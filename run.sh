#!/usr/bin/env bash
# All ROCm env var tuning tested on gfx1151 (ROCm 7.2.0) degraded FPS from
# 27.7 to 25.6. Running with no overrides restores baseline performance.
exec python infer_optical_flow.py "$@"
