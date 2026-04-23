#!/usr/bin/env python3
"""RAFT Large optical-flow inference on a pair of consecutive video frames.

Usage
-----
    python infer_optical_flow.py --video input.mp4
    python infer_optical_flow.py --video input.mp4 --frame 50 --output result.png
    python infer_optical_flow.py --video input.mp4 --resize 520x960
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
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="0-based index of the first frame (second frame is frame+1)",
    )
    p.add_argument(
        "--output",
        default="flow_output.png",
        help="Path for the saved output image (default: flow_output.png)",
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


def preprocess(
    img1: torch.Tensor,
    img2: torch.Tensor,
    transforms,
    resize_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize so dimensions are divisible by 8, then apply model transforms."""
    if resize_hw is not None:
        h, w = resize_hw
    else:
        _, h, w = img1.shape
        h = max(round_down_to_multiple(h, 8), 128)
        w = max(round_down_to_multiple(w, 8), 128)

    img1 = F.resize(img1, [h, w], antialias=False)
    img2 = F.resize(img2, [h, w], antialias=False)

    # Add batch dimension before transforms (expects (B,C,H,W) or (C,H,W))
    return transforms(img1, img2)


def build_composite(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    flow_rgb: torch.Tensor,
) -> np.ndarray:
    """Create a side-by-side [frame1 | frame2 | flow] composite as an HWC uint8 array."""
    panels = []
    for t in (frame1, frame2, flow_rgb):
        t = t.cpu()
        if t.is_floating_point():
            t = ((t + 1.0) / 2.0).clamp(0, 1)
            t = (t * 255).to(torch.uint8)
        panels.append(t.permute(1, 2, 0).numpy())
    return np.concatenate(panels, axis=1)


def main() -> None:
    args = parse_args()

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device : {torch.cuda.get_device_name(0)}  (ROCm/HIP)")
    else:
        print("WARNING: No GPU detected — running on CPU (will be slow).")

    # ---- precision setup ----
    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = dtype_map[args.param_dtype]
    if amp_dtype is not None:
        _patch_corr_block_dtype(amp_dtype)

    # ---- load model ----
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device).eval()
    if args.compile:
        model = torch.compile(model)
    transforms = weights.transforms()
    tags = []
    tags.append("compiled" if args.compile else "eager")
    tags.append(args.param_dtype)
    print(f"Model  : RAFT Large  ({sum(p.numel() for p in model.parameters()):,} params, {', '.join(tags)})")

    # ---- extract frames ----
    rgb1, rgb2 = extract_frames(args.video, args.frame)
    h_orig, w_orig = rgb1.shape[:2]
    print(f"Video  : {args.video}  ({w_orig}x{h_orig})")
    print(f"Frames : {args.frame} and {args.frame + 1}")

    img1 = to_tensor(rgb1)
    img2 = to_tensor(rgb2)

    # ---- preprocess ----
    resize_hw = None
    if args.resize:
        parts = args.resize.split("x")
        resize_hw = (int(parts[0]), int(parts[1]))
    img1_p, img2_p = preprocess(img1, img2, transforms, resize_hw)

    # Ensure batch dimension
    if img1_p.dim() == 3:
        img1_p = img1_p.unsqueeze(0)
        img2_p = img2_p.unsqueeze(0)

    print(f"Input  : {img1_p.shape}  dtype={img1_p.dtype}  range=[{img1_p.min():.2f}, {img1_p.max():.2f}]")

    # ---- inference ----
    img1_d = img1_p.to(device)
    img2_d = img2_p.to(device)

    amp_ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()

    # Warmup pass (compiles HIP kernels on first run)
    with torch.no_grad(), amp_ctx:
        _ = model(img1_d, img2_d)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()

    with torch.no_grad(), amp_ctx:
        list_of_flows = model(img1_d, img2_d)

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    predicted_flow = list_of_flows[-1]  # (1, 2, H, W)
    flow = predicted_flow[0]            # (2, H, W)

    magnitude = flow.norm(dim=0)
    print(f"Latency: {elapsed_ms:.1f} ms  (excluding warmup)")
    print(f"Flow   : shape={tuple(flow.shape)}  "
          f"mag min={magnitude.min():.3f}  max={magnitude.max():.3f}  mean={magnitude.mean():.3f}")

    # ---- visualize ----
    flow_rgb = flow_to_image(flow.cpu())

    # Resize originals to match flow output for the composite
    img1_vis = F.resize(img1_p[0].cpu(), list(flow_rgb.shape[1:]), antialias=False)
    img2_vis = F.resize(img2_p[0].cpu(), list(flow_rgb.shape[1:]), antialias=False)

    composite = build_composite(img1_vis, img2_vis, flow_rgb)
    composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), composite_bgr)
    print(f"Saved  : {out_path.resolve()}  ({composite.shape[1]}x{composite.shape[0]})")


if __name__ == "__main__":
    main()
