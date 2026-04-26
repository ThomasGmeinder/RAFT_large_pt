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
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.models.optical_flow._utils import grid_sample as _grid_sample
from torchvision.models.optical_flow.raft import CorrBlock
from torchvision.utils import flow_to_image


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
    p.add_argument("--video", default="Geisskopf_Gap_Jump.MOV", help="Path to input video file")
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="0-based index of the first frame (second frame is frame+1)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path (default: flow_output.png or flow_output.mp4 with --realtime)",
    )
    p.add_argument(
        "--resize",
        default=None,
        help="Explicit HxW to resize frames to, e.g. 520x960",
    )
    p.add_argument(
        "--compile",
        type=lambda v: v.lower() not in ("0", "false", "no", "off"),
        default=True,
        metavar="BOOL",
        help="Use torch.compile (default: True)",
    )
    p.add_argument(
        "--param_dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Inference precision: fp32, fp16, or bf16 (default: fp16)",
    )
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Process every frame pair and write a side-by-side MP4 video",
    )
    return p.parse_args()


def extract_frames(
    video_path: str, frame_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read two consecutive RGB frames from *video_path* starting at *frame_idx*."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video '{video_path}'")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx + 1 >= total:
        cap.release()
        sys.exit(
            f"ERROR: frame {frame_idx}+1 out of range (video has {total} frames)"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok1, bgr1 = cap.read()
    ok2, bgr2 = cap.read()
    cap.release()

    if not ok1 or not ok2:
        sys.exit(f"ERROR: failed to read frames {frame_idx} and {frame_idx + 1}")

    rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return rgb1, rgb2


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 ndarray -> CHW uint8 tensor."""
    return torch.from_numpy(img).permute(2, 0, 1)


def round_down_to_multiple(value: int, multiple: int) -> int:
    return value - (value % multiple)


def compute_resize_hw(
    h: int, w: int, resize_arg: str | None
) -> tuple[int, int] | None:
    if resize_arg is not None:
        parts = resize_arg.split("x")
        return (int(parts[0]), int(parts[1]))
    h2 = max(round_down_to_multiple(h, 8), 128)
    w2 = max(round_down_to_multiple(w, 8), 128)
    if h2 == h and w2 == w:
        return None
    return (h2, w2)


def preprocess(
    img1: torch.Tensor,
    img2: torch.Tensor,
    transforms,
    resize_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize so dimensions are divisible by 8, then apply model transforms."""
    if resize_hw is not None:
        h, w = resize_hw
        img1 = F.resize(img1, [h, w], antialias=False)
        img2 = F.resize(img2, [h, w], antialias=False)

    return transforms(img1, img2)


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

    for y in ys:
        for x in xs:
            dx = float(flow_np[0, y, x]) * scale
            dy = float(flow_np[1, y, x]) * scale
            arrow_len = np.sqrt(dx * dx + dy * dy)

            if arrow_len < 2.0:
                cv2.circle(canvas, (x, y), 1, (100, 100, 100), -1)
            else:
                brightness = min(arrow_len / (step * 0.8), 1.0)
                r = int(255 * brightness)
                g = int(200 * brightness)
                x2 = int(x + dx)
                y2 = int(y + dy)
                cv2.arrowedLine(canvas, (x, y), (x2, y2), (r, g, 50), 1, tipLength=0.3)

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
        model = torch.compile(model)
    transforms = weights.transforms()

    tags = ["compiled" if args.compile else "eager", args.param_dtype]
    print(f"Model  : RAFT Large  ({sum(p.numel() for p in model.parameters()):,} params, {', '.join(tags)})")

    amp_ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()

    return device, model, transforms, amp_ctx


def run_single_pair(args: argparse.Namespace) -> None:
    """Original single-frame-pair mode."""
    device, model, transforms, amp_ctx = setup_model(args)

    rgb1, rgb2 = extract_frames(args.video, args.frame)
    h_orig, w_orig = rgb1.shape[:2]
    print(f"Video  : {args.video}  ({w_orig}x{h_orig})")
    print(f"Frames : {args.frame} and {args.frame + 1}")

    img1 = to_tensor(rgb1)
    img2 = to_tensor(rgb2)

    resize_hw = compute_resize_hw(h_orig, w_orig, args.resize)
    img1_p, img2_p = preprocess(img1, img2, transforms, resize_hw)

    if img1_p.dim() == 3:
        img1_p = img1_p.unsqueeze(0)
        img2_p = img2_p.unsqueeze(0)

    print(f"Input  : {img1_p.shape}  dtype={img1_p.dtype}  range=[{img1_p.min():.2f}, {img1_p.max():.2f}]")

    img1_d = img1_p.to(device)
    img2_d = img2_p.to(device)

    with torch.no_grad(), amp_ctx:
        _ = model(img1_d, img2_d)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    with torch.no_grad(), amp_ctx:
        list_of_flows = model(img1_d, img2_d)

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    flow = list_of_flows[-1][0]
    magnitude = flow.norm(dim=0)
    print(f"Latency: {elapsed_ms:.1f} ms  (excluding warmup)")
    print(f"Flow   : shape={tuple(flow.shape)}  "
          f"mag min={magnitude.min():.3f}  max={magnitude.max():.3f}  mean={magnitude.mean():.3f}")

    flow_rgb = flow_to_image(flow.cpu())
    img1_vis = F.resize(img1_p[0].cpu(), list(flow_rgb.shape[1:]), antialias=False)
    img2_vis = F.resize(img2_p[0].cpu(), list(flow_rgb.shape[1:]), antialias=False)

    composite = build_composite_4(img1_vis, img2_vis, flow_rgb, flow.cpu())
    composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), composite_bgr)
    print(f"Saved  : {out_path.resolve()}  ({composite.shape[1]}x{composite.shape[0]})")


def run_realtime(args: argparse.Namespace) -> None:
    """Process every consecutive frame pair and write a side-by-side MP4."""
    device, model, transforms, amp_ctx = setup_model(args)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video '{args.video}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pairs = total_frames - 1

    print(f"Video  : {args.video}  ({w_orig}x{h_orig}, {total_frames} frames, {fps:.1f} fps)")

    resize_hw = compute_resize_hw(h_orig, w_orig, args.resize)
    if resize_hw:
        out_h, out_w = resize_hw
    else:
        out_h, out_w = h_orig, w_orig

    out_path = Path(args.output)

    ok, bgr_prev = cap.read()
    if not ok:
        cap.release()
        sys.exit("ERROR: cannot read first frame")
    rgb_prev = cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2RGB)

    # Warmup with first two frames
    ok, bgr_next = cap.read()
    if not ok:
        cap.release()
        sys.exit("ERROR: video has fewer than 2 frames")
    rgb_next = cv2.cvtColor(bgr_next, cv2.COLOR_BGR2RGB)

    t1 = to_tensor(rgb_prev)
    t2 = to_tensor(rgb_next)
    t1_p, t2_p = preprocess(t1, t2, transforms, resize_hw)
    if t1_p.dim() == 3:
        t1_p = t1_p.unsqueeze(0)
        t2_p = t2_p.unsqueeze(0)

    print(f"Input  : {t1_p.shape}  dtype={t1_p.dtype}")
    print(f"Warmup + calibration ...")

    t1_d = t1_p.to(device) if t1_p.dim() == 4 else t1_p.unsqueeze(0).to(device)
    t2_d = t2_p.to(device) if t2_p.dim() == 4 else t2_p.unsqueeze(0).to(device)

    # Warmup (first run triggers torch.compile)
    with torch.no_grad(), amp_ctx:
        _ = model(t1_d, t2_d)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Calibrate: measure 10 iterations to get stable throughput
    n_cal = min(10, total_pairs)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_cal_start = time.perf_counter()
    for _ in range(n_cal):
        with torch.no_grad(), amp_ctx:
            _ = model(t1_d, t2_d)
    torch.cuda.synchronize() if device.type == "cuda" else None
    cal_ms = (time.perf_counter() - t_cal_start) / n_cal * 1000
    out_fps = 1000.0 / cal_ms

    print(f"Throughput: {cal_ms:.1f} ms/pair -> output video at {out_fps:.1f} fps")

    import imageio
    writer = imageio.get_writer(
        str(out_path), fps=out_fps, codec="libx264",
        quality=None, macro_block_size=1,
        output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )

    print(f"Output : {out_path}  ({out_w * 2}x{out_h * 2}, {out_fps:.1f} fps, H.264)")

    # Reset to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, bgr_prev = cap.read()
    rgb_prev = cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2RGB)

    elapsed_total = 0.0
    pair_idx = 0

    print(f"Processing {total_pairs} frame pairs ...")

    while True:
        ok, bgr_next = cap.read()
        if not ok:
            break

        rgb_next = cv2.cvtColor(bgr_next, cv2.COLOR_BGR2RGB)

        t1 = to_tensor(rgb_prev)
        t2 = to_tensor(rgb_next)
        t1_p, t2_p = preprocess(t1, t2, transforms, resize_hw)
        if t1_p.dim() == 3:
            t1_p = t1_p.unsqueeze(0)
            t2_p = t2_p.unsqueeze(0)

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        with torch.no_grad(), amp_ctx:
            list_of_flows = model(t1_p.to(device), t2_p.to(device))

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed_total += time.perf_counter() - t0

        flow = list_of_flows[-1][0]
        flow_cpu = flow.cpu()
        flow_rgb = flow_to_image(flow_cpu)

        f1_vis = F.resize(t1_p[0].cpu(), list(flow_rgb.shape[1:]), antialias=False)
        f2_vis = F.resize(t2_p[0].cpu(), list(flow_rgb.shape[1:]), antialias=False)
        composite = build_composite_4(f1_vis, f2_vis, flow_rgb, flow_cpu)
        writer.append_data(composite)

        pair_idx += 1
        if pair_idx % 100 == 0 or pair_idx == total_pairs:
            avg_ms = (elapsed_total / pair_idx) * 1000
            eta = (total_pairs - pair_idx) * avg_ms / 1000
            print(f"  {pair_idx:5d}/{total_pairs}  avg={avg_ms:.1f} ms/pair  ETA={eta:.0f}s")

        rgb_prev = rgb_next

    writer.close()
    cap.release()

    avg_ms = (elapsed_total / pair_idx) * 1000 if pair_idx > 0 else 0
    print(f"Done   : {pair_idx} pairs, avg {avg_ms:.1f} ms/pair ({1000/avg_ms:.1f} pairs/sec)")
    print(f"Saved  : {out_path.resolve()}")


def main() -> None:
    args = parse_args()

    if args.output is None:
        args.output = "optical_flow_vectors_video.mp4" if args.realtime else "flow_output.png"
    elif args.realtime and args.output.endswith(".png"):
        args.output = args.output.rsplit(".", 1)[0] + ".mp4"

    if args.realtime:
        run_realtime(args)
    else:
        run_single_pair(args)


if __name__ == "__main__":
    main()
