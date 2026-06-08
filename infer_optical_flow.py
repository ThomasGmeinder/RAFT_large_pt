#!/usr/bin/env python3
"""RAFT Large optical-flow inference on a pair of consecutive video frames.

Usage
-----
    python infer_optical_flow.py --video input.mp4
    python infer_optical_flow.py --video input.mp4 --frame 50 --output result.png
    python infer_optical_flow.py --video input.mp4 --realtime --output flow_video.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import shutil
import subprocess

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.models.optical_flow._utils import grid_sample as _grid_sample
from torchvision.models.optical_flow.raft import CorrBlock
from torchvision.utils import flow_to_image


@dataclass
class FlowResult:
    flow: torch.Tensor  # (2, H, W) CPU tensor — raw vector field from inference
    elapsed_s: float    # forward-pass wall time in seconds


def _patch_corr_block_dtype(dtype: torch.dtype) -> None:
    """Monkey-patch CorrBlock so the correlation pyramid and grid_sample run in *dtype*.

    By default autocast does not cover grid_sample or the correlation volume
    storage, leaving them in fp32.  This patch forces both into the target
    dtype, which roughly halves memory-bandwidth pressure in the 12-iteration
    update loop.
    """
    _orig_build = CorrBlock.build_pyramid.__wrapped__ if hasattr(CorrBlock.build_pyramid, "__wrapped__") else CorrBlock.build_pyramid  # noqa: E501
    _orig_index = CorrBlock.index_pyramid.__wrapped__ if hasattr(CorrBlock.index_pyramid, "__wrapped__") else CorrBlock.index_pyramid  # noqa: E501

    def _build(self, fmap1, fmap2):
        _orig_build(self, fmap1.to(dtype), fmap2.to(dtype))
        self.corr_pyramid = [v.to(dtype) for v in self.corr_pyramid]

    def _index(self, centroids_coords):
        centroids_coords = centroids_coords.to(dtype)
        side = 2 * self.radius + 1
        di = torch.linspace(-self.radius, self.radius, side)
        dj = torch.linspace(-self.radius, self.radius, side)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1)
        delta = delta.to(centroids_coords.device, dtype=dtype).view(1, side, side, 2)

        bs, _, h, w = centroids_coords.shape
        centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(bs * h * w, 1, 1, 2)

        indexed = []
        for corr_vol in self.corr_pyramid:
            coords = centroids_coords + delta
            out = _grid_sample(corr_vol, coords, align_corners=True, mode="bilinear")
            indexed.append(out.view(bs, h, w, -1))
            centroids_coords = centroids_coords / 2

        return torch.cat(indexed, dim=-1).permute(0, 3, 1, 2).contiguous()

    _build.__wrapped__ = _orig_build  # type: ignore[attr-defined]
    _index.__wrapped__ = _orig_index  # type: ignore[attr-defined]
    CorrBlock.build_pyramid = _build  # type: ignore[assignment]
    CorrBlock.index_pyramid = _index  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAFT Large optical-flow inference (PyTorch / ROCm)"
    )
    p.add_argument("--video", default="Geisskopf_Gap_Jump.mp4", help="Path to input video file")
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="0-based index of the first frame (second frame is frame+1)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path — video extension (mp4, avi, mov, mkv) processes the whole video; "
             "image extension (png, jpg) processes only --frame. Default: optical_flow_vectors_video.mp4",
    )
    p.add_argument(
        "--resize",
        default="376x672",
        help="HxW to resize frames to (default: 376x672). Use 'none' for native resolution",
    )
    p.add_argument(
        "--compile",
        type=lambda v: v.lower() not in ("0", "false", "no", "off"),
        default=True,
        metavar="BOOL",
        help="Use torch.compile (default: True)",
    )
    p.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead)",
    )
    p.add_argument(
        "--param_dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Inference precision: fp32, fp16, or bf16 (default: fp16)",
    )
    p.add_argument(
        "--vaapi",
        type=lambda v: v.lower() not in ("0", "false", "no", "off"),
        default=True,
        metavar="BOOL",
        help="Use VAAPI hardware H.264 encoder if available (default: True)",
    )
    return p.parse_args()


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 ndarray -> CHW uint8 tensor."""
    return torch.from_numpy(img).permute(2, 0, 1)


def round_down_to_multiple(value: int, multiple: int) -> int:
    return value - (value % multiple)


def compute_resize_hw(
    h: int, w: int, resize_arg: str | None
) -> tuple[int, int] | None:
    """Return (H, W) to resize to, or None if no resize is needed."""
    if resize_arg is not None and resize_arg.lower() not in ("none", ""):
        parts = resize_arg.split("x")
        return (int(parts[0]), int(parts[1]))
    h2 = max(round_down_to_multiple(h, 8), 128)
    w2 = max(round_down_to_multiple(w, 8), 128)
    if h2 == h and w2 == w:
        return None
    return (h2, w2)


def preprocess_one(
    img: torch.Tensor,
    transforms,
    resize_hw: tuple[int, int] | None,
) -> torch.Tensor:
    """Resize + normalize a single frame. Returns (3, H, W) float tensor.

    Calling transforms(img, img) and discarding the duplicate output is safe
    because RAFT transforms apply the same normalization to each image
    independently.
    """
    if resize_hw is not None:
        img = F.resize(img, list(resize_hw), antialias=False)
    out, _ = transforms(img, img)
    return out


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """CHW tensor (float [-1,1] or uint8) -> HWC uint8 numpy array."""
    t = t.cpu()
    if t.is_floating_point():
        t = ((t + 1.0) / 2.0).clamp(0, 1)
        t = (t * 255).to(torch.uint8)
    return t.permute(1, 2, 0).numpy()


def draw_flow_vectors(
    flow: torch.Tensor, bg: np.ndarray, step: int = 16, scale: float = 5.0
) -> np.ndarray:
    """Draw actual predicted flow vectors on top of *bg* (HWC uint8 RGB).

    *flow* is (2, H, W) with horizontal/vertical displacement in pixels.
    Arrows are drawn on a sub-sampled grid with *step*-pixel spacing.
    *scale* amplifies the vectors for visibility (1.0 = 1 pixel of flow
    draws 1 pixel of arrow length).  Vectors shorter than 2 pixels on
    screen are drawn as dots.
    """
    flow_np = flow.cpu().float().numpy()
    h, w = flow_np.shape[1], flow_np.shape[2]
    canvas = bg.copy()

    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)

    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # (ny, nx) grid coordinates

    # All flow values and magnitudes in one numpy pass
    gx = flow_np[0][np.ix_(ys, xs)] * scale   # (ny, nx)
    gy = flow_np[1][np.ix_(ys, xs)] * scale
    lengths = np.hypot(gx, gy)
    threshold = step * 0.8

    # Dots: single bulk numpy write — no Python loop
    dot = lengths < 2.0
    canvas[yy[dot], xx[dot]] = (100, 100, 100)

    # Arrows: pre-compute all parameters as flat Python lists, then tight cv2 loop
    amask = ~dot
    if amask.any():
        brightness = np.clip(lengths[amask] / threshold, 0.0, 1.0)
        x0s = xx[amask].tolist()
        y0s = yy[amask].tolist()
        x1s = (xx[amask] + gx[amask]).astype(int).tolist()
        y1s = (yy[amask] + gy[amask]).astype(int).tolist()
        rs  = (255 * brightness).astype(np.uint8).tolist()
        gs  = (200 * brightness).astype(np.uint8).tolist()
        for x0, y0, x1, y1, r, g in zip(x0s, y0s, x1s, y1s, rs, gs):
            cv2.arrowedLine(canvas, (x0, y0), (x1, y1), (r, g, 50), 1, tipLength=0.3)

    return canvas


def build_composite_4(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    flow_rgb: torch.Tensor,
    flow: torch.Tensor,
) -> np.ndarray:
    """2x2 grid: [frame1 | frame2] / [flow color | flow vectors]."""
    f1 = _tensor_to_uint8(frame1)
    f2 = _tensor_to_uint8(frame2)
    fc = _tensor_to_uint8(flow_rgb)
    fv = draw_flow_vectors(flow, f1.copy())
    top = np.concatenate([f1, f2], axis=1)
    bot = np.concatenate([fc, fv], axis=1)
    return np.concatenate([top, bot], axis=0)


class VaapiWriter:
    """Pipe RGB frames to ffmpeg using AMD VAAPI hardware H.264 encoding.

    Offloads encoding to the VCN hardware block, freeing the CPU and reducing
    DRAM pressure between GPU inference calls on the iGPU.
    """

    VAAPI_DEVICE = "/dev/dri/renderD128"

    def __init__(self, path: str, fps: float, width: int, height: int):
        cmd = [
            "ffmpeg", "-y",
            "-vaapi_device", self.VAAPI_DEVICE,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi", "-qp", "18",
            path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def append_data(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("ffmpeg") is not None and os.path.exists(cls.VAAPI_DEVICE)


def setup_model(args: argparse.Namespace):
    """Shared setup: device, precision, model, transforms, autocast context."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device : {torch.cuda.get_device_name(0)}  (ROCm/HIP)")
    else:
        print("WARNING: No GPU detected — running on CPU (will be slow).")

    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = dtype_map[args.param_dtype]
    if amp_dtype is not None:
        _patch_corr_block_dtype(amp_dtype)

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device).eval()
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    transforms = weights.transforms()

    tags = [f"compiled({args.compile_mode})" if args.compile else "eager", args.param_dtype]
    print(f"Model  : RAFT Large  ({sum(p.numel() for p in model.parameters()):,} params, {', '.join(tags)})")

    amp_ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()

    return device, model, transforms, amp_ctx


def run(args: argparse.Namespace, whole_video: bool) -> None:
    """Unified inference loop for single-pair (PNG) and whole-video (MP4) modes."""
    device, model, transforms, amp_ctx = setup_model(args)
    out_path = Path(args.output)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video '{args.video}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = 0 if whole_video else args.frame
    if start_frame + 1 >= total_frames:
        cap.release()
        sys.exit(f"ERROR: frame {start_frame}+1 out of range ({total_frames} frames)")

    total_pairs = (total_frames - 1 - start_frame) if whole_video else 1
    resize_hw = compute_resize_hw(h_orig, w_orig, args.resize)
    out_h, out_w = resize_hw if resize_hw else (h_orig, w_orig)

    print(f"Video  : {args.video}  ({w_orig}x{h_orig}, {total_frames} frames, {src_fps:.1f} fps)")
    if whole_video:
        print(f"Frames : 0 .. {total_frames - 1}")
    else:
        print(f"Frames : {start_frame} and {start_frame + 1}")
    if resize_hw:
        print(f"Resize : {w_orig}x{h_orig} -> {out_w}x{out_h}")

    # Read first two frames for warmup (do not count toward inference)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, bgr1 = cap.read()
    ok2, bgr2 = cap.read()
    if not ok or not ok2:
        cap.release()
        sys.exit("ERROR: cannot read warmup frames")
    t1_warm = preprocess_one(to_tensor(cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)), transforms, resize_hw).unsqueeze(0)
    t2_warm = preprocess_one(to_tensor(cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)), transforms, resize_hw).unsqueeze(0)

    print(f"Input  : {t1_warm.shape}  dtype={t1_warm.dtype}")

    # Pre-allocate fixed-address GPU input buffers used by warmup, calibration,
    # AND the main loop — consistent pointers let reduce-overhead CUDA graphs
    # replay without extra input copies on every call.
    t1_buf = t1_warm.to(device)
    t2_buf = t2_warm.to(device)

    print("Warmup ...")
    for _ in range(3):
        with torch.no_grad(), amp_ctx:
            _ = model(t1_buf, t2_buf)
    torch.cuda.synchronize() if device.type == "cuda" else None

    if whole_video:
        # Calibrate using the same fixed buffers as the main loop
        n_cal = min(10, total_pairs)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_cal = time.perf_counter()
        for _ in range(n_cal):
            with torch.no_grad(), amp_ctx:
                _ = model(t1_buf, t2_buf)
        torch.cuda.synchronize() if device.type == "cuda" else None
        cal_ms = (time.perf_counter() - t_cal) / n_cal * 1000
        out_fps = min(src_fps, 1000.0 / cal_ms)
        print(f"Calibration: {cal_ms:.1f} ms/pair ({1000.0/cal_ms:.1f} fps) -> output {out_fps:.1f} fps")

        composite_w, composite_h = out_w * 2, out_h * 2
        use_vaapi = args.vaapi and VaapiWriter.is_available()
        if use_vaapi:
            writer = VaapiWriter(str(out_path), out_fps, composite_w, composite_h)
            print(f"Output : {out_path}  ({composite_w}x{composite_h}, {out_fps:.1f} fps, H.264/VAAPI)")
        else:
            import imageio
            writer = imageio.get_writer(
                str(out_path), fps=out_fps, codec="libx264",
                quality=None, macro_block_size=1,
                output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
            )
            print(f"Output : {out_path}  ({composite_w}x{composite_h}, {out_fps:.1f} fps, H.264/libx264)")

    # Reset to start; preprocess first frame and load into t1_buf (fixed GPU address)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, bgr_prev = cap.read()
    t1_p = preprocess_one(to_tensor(cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2RGB)), transforms, resize_hw).unsqueeze(0)
    t1_buf.copy_(t1_p)  # load into pre-allocated buffer — same GPU address as warmup

    elapsed_total = 0.0
    pair_idx = 0
    t_wall = time.perf_counter()
    print(f"Processing {total_pairs} pair{'s' if total_pairs > 1 else ''} ...")

    composite = result = None  # satisfy reference-before-assignment for single-pair path
    while pair_idx < total_pairs:
        ok, bgr_next = cap.read()
        if not ok:
            break

        # Preprocess only the new frame; t1_p cached from previous iteration (CPU side for composite)
        t2_p = preprocess_one(to_tensor(cv2.cvtColor(bgr_next, cv2.COLOR_BGR2RGB)), transforms, resize_hw).unsqueeze(0)
        t2_buf.copy_(t2_p)  # load into fixed buffer — CUDA graph sees consistent pointers

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        with torch.no_grad(), amp_ctx:
            list_of_flows = model(t1_buf, t2_buf)

        torch.cuda.synchronize() if device.type == "cuda" else None
        result = FlowResult(flow=list_of_flows[-1][0].cpu(), elapsed_s=time.perf_counter() - t0)
        elapsed_total += result.elapsed_s

        # Post-processing: t1_p[0] and t2_p[0] are already at the same spatial
        # size as flow_rgb — no F.resize needed
        flow_rgb = flow_to_image(result.flow)
        composite = build_composite_4(t1_p[0], t2_p[0], flow_rgb, result.flow)

        if whole_video:
            writer.append_data(composite)

        t1_buf, t2_buf = t2_buf, t1_buf  # swap fixed buffers — t1 stays on GPU, no re-upload
        t1_p = t2_p  # keep CPU copy in sync for composite building
        pair_idx += 1

        if whole_video and (pair_idx % 100 == 0 or pair_idx == total_pairs):
            avg_ms = elapsed_total / pair_idx * 1000
            eta = (total_pairs - pair_idx) * avg_ms / 1000
            print(f"  {pair_idx:5d}/{total_pairs}  avg={avg_ms:.1f} ms/pair  ETA={eta:.0f}s")

    cap.release()

    avg_ms = elapsed_total / pair_idx * 1000 if pair_idx > 0 else 0
    infer_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    if whole_video:
        writer.close()
        wall_s = time.perf_counter() - t_wall
        print(f"Summary: {pair_idx} pairs  total={wall_s:.1f}s  avg={avg_ms:.1f} ms/pair  FPS={infer_fps:.1f}")
    else:
        magnitude = result.flow.norm(dim=0)
        print(f"Summary: 1 pair  time={avg_ms:.1f} ms  FPS={infer_fps:.1f}")
        print(f"Flow   : shape={tuple(result.flow.shape)}  "
              f"mag min={magnitude.min():.3f}  max={magnitude.max():.3f}  mean={magnitude.mean():.3f}")
        cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    print(f"Saved  : {out_path.resolve()}")


def main() -> None:
    args = parse_args()

    _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    if args.output is None:
        args.output = "optical_flow_vectors_video.mp4"
        whole_video = True
    else:
        ext = Path(args.output).suffix.lower()
        if ext in _VIDEO_EXTS:
            whole_video = True
        elif ext in _IMAGE_EXTS:
            whole_video = False
        else:
            sys.exit(f"ERROR: unrecognized output extension '{ext}'. Use a video ({', '.join(sorted(_VIDEO_EXTS))}) or image ({', '.join(sorted(_IMAGE_EXTS))}) extension.")

    run(args, whole_video)


if __name__ == "__main__":
    main()
