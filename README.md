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
| `--compile False --param_dtype bf16` | 69 ms | ~14.4 |
| `--compile True --param_dtype fp32` | 53 ms | ~18.7 |
| Default (`--compile True --param_dtype bf16`) | 50 ms | ~20.0 (1.35x) |

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
| `--output` | `flow_output.png` | Path for the saved composite image |
| `--resize` | auto | Explicit `HxW` (e.g. `520x960`); if omitted, dims are rounded down to a multiple of 8 |
| `--compile` | `True` | `torch.compile` for faster inference (set `False` to disable) |
| `--param_dtype` | `bf16` | Inference precision: `fp32`, `fp16`, or `bf16` |

### Example output

The script saves a side-by-side PNG: **frame 1 | frame 2 | flow visualization**.

```
Device : AMD Radeon Graphics  (ROCm/HIP)
Model  : RAFT Large  (5,257,536 params, compiled, bf16)
Video  : input.mp4  (672x376)
Frames : 0 and 1
Latency: 50.0 ms  (excluding warmup)
Saved  : /path/to/flow_output.png  (2016x376)
```
