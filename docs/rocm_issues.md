# ROCm issues on this machine

gfx1151 (Ryzen AI MAX+ 395 / Radeon 8060S), Ubuntu 24.04, kernel
`6.17.0-1032-oem`. The system ROCm install is **7.2.4**
(`/opt/rocm` → `/opt/rocm-7.2.4`, `rocm-core 7.2.4.70204`).
`setup_venv.sh` can also build a venv for 7.2.1, 7.2.4, or 10.0.0 wheels.

RAFT at 672×376, fp16, `torch.compile` mode `reduce-overhead`, is the
comparison point. The documented ROCm 7.2.1 result is **29 ms/pair, 34.5 fps**.
The script times a forward with `time.perf_counter()` and
`torch.cuda.synchronize()`. The full-video summary also includes copying the
flow tensor back to the CPU. The calibration line does not.

GPU performance level on this machine is `auto`
(`power_dpm_force_performance_level=auto`, `power_dpm_state=performance`).
The CPU governor is `performance`. That is not a locked-low clock.

## `/tmp` is mounted `noexec`

**Symptom.** `torch.compile` dies while loading Triton's `hip_utils` module:

```text
ImportError: .../hip_utils....so: failed to map segment from shared object
```

**Cause.** `/tmp` is `tmpfs` with `noexec` (`findmnt -no OPTIONS /tmp`). Triton
and Inductor write the compiled `.so` under `/tmp` and then `dlopen` it.

**Resolution.** `infer_optical_flow.py` points `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` at `~/.cache` before importing torch. This applies to every
ROCm version. An exported value still wins, because the script uses
`os.environ.setdefault`.

## ROCm 10 GPU event timer

**Symptom.** A plain fp16 `conv2d` raises `RuntimeError: miopenStatusUnknownError`.
With `MIOPEN_LOG_LEVEL=5` every solver fails inside `EvaluateInvokers`:

```text
Invalid elapsed time detected in EvaluateInvokers, failed condition: elapsed <= 0
No suitable algorithm was found to execute the required convolution
```

A 4096×4096 fp16 GEMM that takes about **87 ms** of wall time is reported by
`torch.cuda.Event.elapsed_time` as about **0.009–0.018 ms**. `max-autotune`
then ranks Triton GEMMs at about **−173,000,000 ms** and cannot choose a kernel.

**What this is not.**

- It is not the RAFT source. The fp16 correlation / `grid_sample` patch, the
  fixed GPU buffers, and `reduce-overhead` are the same code that produced 29 ms
  on ROCm 7.2.1.
- It is not a host-versus-wheel library mismatch. After `import torch` from the
  ROCm 10.0.0 wheel (`torch 2.13.0+rocm10.0.0`, HIP build `7.15.26333`),
  `libamdhip64`, `libhsa-runtime64`, and `libMIOpen` are mapped from the venv.
  The only host library in that process was `libhsa-amd-aqlprofile64` from
  `/opt/rocm-7.2.4`, via `ldconfig`.
- Putting the wheel copy of `libhsa-amd-aqlprofile64` first on
  `LD_LIBRARY_PATH` removed the host library from the map. The event timer stayed
  broken (0.009 ms versus 87 ms). Preloading both copies made the event time a
  large negative number.
- Disabling Winograd Fury
  (`MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3=0` and `F3X2=0`) and leaving
  `MIOPEN_FIND_MODE` unset still rejected every solver, including implicit GEMM
  and the naive kernel. There is no gfx1151 system FindDB in the wheel or under
  `/opt/rocm`.

**Resolution in use.** For an installed torch version containing `+rocm10.`,
the script sets `MIOPEN_FIND_MODE=2` (FAST) before torch is imported. FAST skips
the benchmark and uses an immediate fallback kernel. That is enough to run, and
it is slower than a tuned kernel. ROCm 7.2.1 and 7.2.4 do not set this variable.

Forcing a single solver with `MIOPEN_DEBUG_FIND_ONLY_SOLVER` under the default
Find mode still failed the elapsed-time check. Under FAST, several named solvers
on one RAFT-sized fp16 conv were all about 0.06–0.10 ms, so naming one solver
did not recover the old end-to-end latency.

## Measured latency

Same video (`Geisskopf_Gap_Jump.mp4`), 672×376, fp16, `reduce-overhead`, unless
noted. "Forward" is the calibration or a synced loop around `model(...)` only.
"Summary" is the script's per-pair average, which also copies the result to the CPU.

| Stack | Forward | Summary |
|---|---|---|
| ROCm 7.2.1, documented, default Find | 28.1 ms (35.6 fps) | 29.0 ms (34.5 fps) |
| ROCm 7.2.4 wheels, this machine, 2026-09-23, FAST still forced for every version | 40.0 ms (25.0 fps) | 52.6 ms (19.0 fps) |
| ROCm 7.2.4 wheels, this machine, 2026-09-23, default Find, pmode already `performance` | 35.3 ms (28.3 fps) | 38.9 ms (25.7 fps) |
| ROCm 10.0.0 wheels, FAST | 44.0 ms calibration; 44.8 ms mean over 15 iters | 60.9 ms (16.4 fps) |

ROCm 10 forward-only, 15 iterations after warmup, still in FAST mode:

| Compile | Min | Mean | Max |
|---|---|---|---|
| Eager | 55.4 ms | 57.0 ms | 59.4 ms |
| `default` | 50.0 ms | 51.7 ms | 55.2 ms |
| `reduce-overhead` | 44.0 ms | 44.8 ms | 47.3 ms |
| `max-autotune` | not measured | | |

`reduce-overhead` is still faster than eager on ROCm 10. The missing time versus
29 ms is the convolution kernel MIOpen will not select while the event timer is
wrong. `max-autotune` was stopped after about 15 minutes because its rankings
used the broken negative event times.

Leaving Find at its default on 7.2.4 improved the summary from 52.6 ms to
38.9 ms. That is the 7.2.4 result for this script. It is still slower than the
documented 7.2.1 result of 29.0 ms. An older note in this repo, from an earlier
revision of the script, was that 7.2.4 wheels alone dropped fps from 27.7 to
25.0 versus 7.2.1. GPU `power_dpm_state` was already `performance` for the
38.9 ms run, so that gain is from MIOpen Find, not from changing pmode.
