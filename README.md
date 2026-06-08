# RAFT Large Optical Flow Inference (ROCm)

Predict optical flow between consecutive video frames using
[RAFT Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)
on an AMD GPU via PyTorch + ROCm.

Tested on **AMD Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, gfx1151)**
with ROCm 7.2.1 on Ubuntu 24.04.

**Performance** at 672×376 on the integrated GPU:

| Configuration | Latency | FPS |
|---------------|---------|-----|
| `--compile False --param_dtype fp32` | 67 ms | ~14.8 |
| `--compile True --param_dtype fp32` | 53 ms | ~18.7 (1.26×) |
| `--compile True --param_dtype bf16` | 51 ms | ~19.6 (1.32×) |
| compiled fp16, no CorrBlock patch | 50 ms | ~20.1 (1.34×) |
| `--compile True --param_dtype fp16` (`--compile-mode default`) | 34 ms | ~29.7 (1.97×) |
| **Default** (`--compile-mode reduce-overhead --param_dtype fp16`) | **~29 ms** | **~34.5 (2.33×)** |

`--compile-mode reduce-overhead` uses HIP graph capture to replay RAFT's
12-iteration GRU loop with near-zero kernel launch overhead, adding ~4–5 FPS
over the default compile mode.

fp16/bf16 autocast alone barely helps because `grid_sample` — called 48
times per forward pass — is not in autocast's promotion list and stays in fp32.
The script patches the RAFT correlation block to force `grid_sample` and the
correlation pyramid into the target dtype, which is where the real speedup
comes from.

**Precision accuracy** — end-point error (EPE) vs the fp32 baseline:

| Metric | fp16 | bf16 |
|--------|------|------|
| Mean EPE | 0.019 px | 0.272 px |
| Median EPE | 0.016 px | 0.252 px |
| 95th percentile EPE | 0.039 px | 0.507 px |
| Max EPE | 0.102 px | 0.967 px |
| Pixels with EPE < 0.1 px | 99.98% | 5.24% |

fp16 is nearly lossless — its mean error is ~100× smaller than the model's own
prediction error on standard benchmarks (1.8–3.1 EPE). bf16 is 14× worse than
fp16 because its 7-bit mantissa (vs fp16's 10-bit) cannot represent the small
coordinate deltas that accumulate across RAFT's 12-iteration GRU loop. fp16 is
the better choice for RAFT: both faster (~29 ms vs 51 ms) and far more accurate.

### Why fp16 is faster than bf16 on RDNA 3.5

Microbenchmarks on gfx1151 reveal that fp16 and bf16 do not perform equally
across all operation types:

| Operation (RAFT-sized) | fp32 | fp16 | bf16 |
|------------------------|------|------|------|
| Element-wise FMA | 30 us | 17 us (1.75×) | 17 us (1.74×) |
| grid_sample | 1029 us | 780 us (1.32×) | 782 us (1.32×) |
| matmul | 179 us | 18 us (10×) | 19 us (9.2×) |
| conv2d 3×3 (128ch, 47×84) | 75 us | 47 us (1.58×) | 84 us (0.89×) |

Element-wise ops, grid_sample, and matmul are equally fast in fp16 and bf16.
However, bf16 conv2d on small feature maps is **slower than fp32** — the
MIOpen kernel library on ROCm 7.2 selects a suboptimal code path for bf16 at
this tensor size. Since RAFT runs many small convolutions in its 12-iteration
GRU loop, this regression erases any bandwidth savings from smaller tensors.

## Setup

```bash
bash setup_venv.sh          # creates .venv, downloads ROCm 7.2.1 wheels
source .venv/bin/activate
```

## Input video

Place any MP4 (or other OpenCV-readable) video file in this directory.
For example, download a free sample clip from
[Pexels](https://www.pexels.com/search/videos/) or use your own recording.

## Run

The output format is inferred from the file extension:

- **Video extension** (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) — processes every
  consecutive frame pair and writes a 2×2 composite H.264 video.
- **Image extension** (`.png`, `.jpg`, etc.) — processes only the single frame pair
  at `--frame` and saves a PNG.
- **No `--output`** — defaults to `optical_flow_vectors_video.mp4` (whole video).

```bash
# Whole video (default)
python infer_optical_flow.py --video input.mp4

# Whole video, explicit output
python infer_optical_flow.py --video input.mp4 --output flow.mp4

# Single frame pair → PNG
python infer_optical_flow.py --video input.mp4 --frame 42 --output frame42.png
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | `Geisskopf_Gap_Jump.mp4` | Path to the input video file |
| `--frame` | `0` | 0-based index of the first frame (second frame is `frame + 1`); single-pair mode only |
| `--output` | `optical_flow_vectors_video.mp4` | Output path — video extension → whole video; image extension → single pair |
| `--resize` | `376x672` | `HxW` to resize frames before inference; `none` for native resolution |
| `--compile` | `True` | Enable `torch.compile` |
| `--compile-mode` | `reduce-overhead` | torch.compile mode: `default`, `reduce-overhead`, `max-autotune` |
| `--param_dtype` | `fp16` | Inference precision: `fp32`, `fp16`, or `bf16` |
| `--vaapi` | `True` | Use AMD VAAPI hardware H.264 encoder if available; falls back to libx264 |

### Full-video mode

Processes all consecutive frame pairs and writes a 2×2 H.264 MP4:

![Sample output frame](docs/sample_output.png)

| Top-left | Top-right | Bottom-left | Bottom-right |
|----------|-----------|-------------|--------------|
| Frame N | Frame N+1 | Flow color wheel | Vector field overlay |

The output video is encoded at the measured inference throughput so it plays
back in real time. H.264 encoding uses the AMD VCN hardware block via VAAPI
by default (`--vaapi True`), keeping the CPU free and avoiding memory bus
contention with GPU inference on the iGPU.

```bash
python infer_optical_flow.py --video Geisskopf_Gap_Jump.mp4 --output optical_flow_vectors_video.mp4
```

```
Device : AMD Radeon Graphics  (ROCm/HIP)
Model  : RAFT Large  (5,257,536 params, compiled(reduce-overhead), fp16)
Video  : Geisskopf_Gap_Jump.mp4  (1280x720, 192 frames, 30.0 fps)
Frames : 0 .. 191
Resize : 1280x720 -> 672x376
Input  : torch.Size([1, 3, 376, 672])  dtype=torch.float32
Warmup ...
Calibration: 28.1 ms/pair (35.6 fps) -> output 30.0 fps
Output : optical_flow_vectors_video.mp4  (1344x752, 30.0 fps, H.264/VAAPI)
Processing 191 image pairs ...
    100/191  avg=28.8 ms/pair  ETA=3s
    191/191  avg=29.0 ms/pair  ETA=0s
Summary: 191 pairs  total=7.9s  avg=29.0 ms/pair  FPS=34.5
Saved  : optical_flow_vectors_video.mp4
```

**Reading the flow visualization:** the color wheel encodes direction; brightness
encodes speed.

- **Camera panning:** uniform color across the whole frame (all pixels move
  together in the same direction).
- **Moving objects:** distinct colored regions with sharp edges at object
  boundaries, standing out against the background.
- **Camera stationary:** noisy, multi-colored pattern. This is normal — RAFT
  detects sub-pixel displacements from sensor noise and compression artifacts.
  Motion magnitudes are very small (< 1 pixel) but the color wheel amplifies
  their random directions.
