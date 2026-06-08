#!/usr/bin/env bash
# ROCm/iGPU performance tuning for gfx1151 (Strix Halo, RDNA 3.5)
# These must be set before Python starts — the HSA/HIP runtime reads them at init.

# Unified memory: blit kernels outperform SDMA on APUs (no PCIe bottleneck)
export HSA_ENABLE_SDMA=0

# Unlock full GTT pool for PyTorch allocations
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100

# Prevent GC pauses mid-loop on a fixed-shape inference workload
export PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.9"

# Cache compiled Inductor/Triton kernels across script launches
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

# MIOpen: use fully-tuned kernels once FindDb cache is populated
export MIOPEN_FIND_MODE=3

# Prefer hipBLASLt over rocBLAS for GEMM (better heuristics for GRU shapes)
export ROCBLAS_USE_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1

# gfx1151 (Strix Halo) specific: disabling SVM has been reported to give a
# large speedup on this APU by removing page-fault overhead on the data path.
# Comment out if you observe crashes or incorrect output.
export HSA_USE_SVM=0

exec python infer_optical_flow.py "$@"
