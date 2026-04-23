# RAFT Large Optical Flow Inference (ROCm)

Predict optical flow between two consecutive video frames using
[RAFT Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)
on an AMD GPU via PyTorch + ROCm.

Tested on **AMD Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, gfx1151)**
with ROCm 7.2 on Ubuntu 24.04.

**Performance** at 672x376 on the integrated GPU:

| Mode | Latency | Frame pairs/sec |
|------|---------|-----------------|
| `--compile False --param_dtype fp32` | 67 ms | ~14.8 |
| `--compile True --param_dtype fp32` | 53 ms | ~18.7 (1.26x) |
| `--compile True --param_dtype bf16` | 51 ms | ~19.6 (1.32x) |
| compiled fp16 autocast only (no corr patch) | 50 ms | ~20.1 (1.34x) |
| Default (`--compile True --param_dtype fp16`) | 34 ms | ~29.7 (1.97x) |

fp16/bf16 with autocast alone barely helps because `grid_sample` -- called 48
times per forward pass -- is not in autocast's promotion list and stays in fp32.
The script patches the RAFT correlation block to force `grid_sample` and the
correlation pyramid into the target dtype, which is where the real speedup
comes from.

**Precision accuracy** -- end-point error (EPE) vs the fp32 baseline:

| Metric | fp16 | bf16 |
|--------|------|------|
| Mean EPE | 0.019 px | 0.272 px |
| Median EPE | 0.016 px | 0.252 px |
| 95th percentile EPE | 0.039 px | 0.507 px |
| Max EPE | 0.102 px | 0.967 px |
| Pixels with EPE < 0.1 px | 99.98% | 5.24% |

fp16 is nearly lossless -- its mean error is ~100x smaller than the model's own
prediction error on standard benchmarks (1.8-3.1 EPE). bf16 is 14x worse than
fp16 because its 7-bit mantissa (vs fp16's 10-bit) cannot represent the small
coordinate deltas that accumulate across RAFT's 12-iteration GRU loop. fp16 is
the better choice for RAFT: both faster (34 ms vs 51 ms) and far more accurate.

### Why fp16 is faster than bf16 on RDNA 3.5

Microbenchmarks on gfx1151 reveal that fp16 and bf16 do not perform equally
across all operation types:

| Operation (RAFT-sized) | fp32 | fp16 | bf16 |
|------------------------|------|------|------|
| Element-wise FMA | 30 us | 17 us (1.75x) | 17 us (1.74x) |
| grid_sample | 1029 us | 780 us (1.32x) | 782 us (1.32x) |
| matmul | 179 us | 18 us (10x) | 19 us (9.2x) |
| conv2d 3x3 (128ch, 47x84) | 75 us | 47 us (1.58x) | 84 us (0.89x) |

Element-wise ops, grid_sample, and matmul are equally fast in fp16 and bf16.
However, bf16 conv2d on small feature maps is **slower than fp32** -- the
MIOpen kernel library on ROCm 7.2 selects a suboptimal code path for bf16 at
this tensor size. Since RAFT runs many small convolutions in its 12-iteration
GRU loop, this regression erases any bandwidth savings from smaller tensors.

**How this was measured:** each operation was run in isolation on the GPU using
tensor sizes matching the RAFT model's internal feature maps. A warmup of 50
iterations was followed by 1000 timed iterations, with `torch.cuda.synchronize()`
before and after to ensure all GPU work completes before reading the wall clock.
The end-to-end RAFT benchmarks (performance table above) used 50 timed iterations
after a 5-iteration warmup.  All numbers are from a single-run session on the
hardware listed at the top of this file.

## Setup

```bash
bash setup_venv.sh          # creates .venv, downloads ROCm 7.2.1 wheels
source .venv/bin/activate
```

## Input video

Place any MP4 (or other OpenCV-readable) video file in this directory.
For example, download one of the free sample clips from
[Pexels](https://www.pexels.com/search/videos/) or use your own recording.

## Run

```bash
python infer_optical_flow.py --video input.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | *(required)* | Path to the input video file |
| `--frame` | `0` | 0-based index of the first frame; second frame is `frame + 1` |
| `--output` | `.png` / `.mp4` | Output path (auto-selects format based on `--realtime`) |
| `--resize` | auto | Explicit `HxW` (e.g. `520x960`); if omitted, dims are rounded down to a multiple of 8 |
| `--compile` | `True` | `torch.compile` for faster inference (set `False` to disable) |
| `--param_dtype` | `fp16` | Inference precision: `fp32`, `fp16`, or `bf16` |
| `--realtime` | off | Process every frame pair and write a side-by-side MP4 video |

### Single-pair mode (default)

Saves a side-by-side PNG: **frame 1 | frame 2 | flow visualization**.

```
python infer_optical_flow.py --video input.mp4
```

### Full-video mode (`--realtime`)

Processes all consecutive frame pairs and writes a side-by-side H.264 MP4:
**original frame (left) | flow visualization (right)**.

The output video is encoded at the measured inference throughput so it plays
back in real time (1 second of video = 1 second of processing).

```bash
python infer_optical_flow.py --video input.mp4 --realtime --output flow_video.mp4
```

```
Device : AMD Radeon Graphics  (ROCm/HIP)
Model  : RAFT Large  (5,257,536 params, compiled, fp16)
Video  : input.mp4  (672x376, 1782 frames, 92.2 fps)
Throughput: 33.5 ms/pair -> output video at 29.9 fps
Processing 1781 frame pairs ...
Done   : 1781 pairs, avg 36.8 ms/pair (27.2 pairs/sec)
Saved  : /path/to/flow_video.mp4
```

**Reading the flow visualization:** the right panel uses the standard optical
flow color wheel -- each color represents a direction of motion, and brightness
represents speed.  What to expect:

- **Camera panning/translating:** a uniform color across the whole frame (all
  pixels move together in the same direction).
- **Moving objects:** distinct colored regions that stand out against the
  background, with sharp edges at object boundaries.
- **Camera stationary:** a noisy, multi-colored pattern.  This is normal --
  RAFT detects sub-pixel displacements from sensor noise and compression
  artifacts.  The motion magnitudes are very small (< 1 pixel) but the color
  wheel amplifies their random directions.
