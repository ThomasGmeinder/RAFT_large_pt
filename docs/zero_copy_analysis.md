# CPU ↔ iGPU copies on Strix Halo

Measured on the Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), Ubuntu 24.04,
`torch 2.13.0+rocm10.0.0`, RAFT Large, fp16, `torch.compile` mode
`reduce-overhead`, frames resized to 376×672. The machine was idle (load
average about 0).

The extra mapped-memory path is not worth the code. On this APU a
`hipMemcpy` is already a DRAM-to-DRAM copy, and for these tensors it finishes
in tens of microseconds. A full video of 191 pairs took 26.1 ms/pair either
way (38.3 FPS, about 7.2–7.7 s wall time, VAAPI encode).

Against a discrete GPU the same copy is faster, but not by a large factor.
The measured round trip (frame up, flow back) is 80 µs. A PCIe 4.0 x16 link,
the usual desktop connection, would take about 200 µs at the throughput
`cudaMemcpy` typically reaches there, so the shared-memory copy is about
**2.5× faster**. It is about 1.3× faster than PCIe 5.0 x16, and about 5×
faster than PCIe 4.0 x8 or PCIe 3.0 x16. None of those gaps is visible next to
a 26 ms forward, which is why the script keeps the ordinary `copy_` and
`.cpu()` calls.

## The copies are small next to the forward

| Step | Size | Time |
|---|---|---|
| New frame, host to device, one synchronized `copy_` | 3.03 MB, float32, 1×3×376×672 | 46 µs |
| Flow, device to host, `.cpu()` | 2.02 MB, float32, 2×376×672 | 34 µs |
| Forward, inputs in ordinary device memory | | 25.8 ms |
| Forward, inputs in coarse-grain mapped memory | | 25.1 ms |
| Forward, inputs in fine-grain mapped memory | | 25.1 ms |

The timed region of `infer_optical_flow.py` (forward, then the flow copy back)
was 25.03 ms with `hipMemcpy` and 25.16 ms with the mapped buffer. Publishing
the preprocessed frame into the mapped pages was another 46 µs, in the same
place as the host-to-device copy.

Strix Halo has no separate video memory. `hipMemcpy` copies 3 MB from one part
of LPDDR5X to another, at about 60–66 GB/s for these sizes. The encoder then
writes feature maps into ordinary device allocations, and the 12-iteration
update loop reads those maps, not the original frame. Removing the frame copy
does not touch the loop that dominates the 26 ms.

A difference of 0.1 ms is also smaller than the run-to-run drift of this chip.
An earlier gap of 1–2 ms, taken while the machine was busy, was that drift.

## Kernel dispatch is a separate cost, and the benchmark already removes it

Eager mode asks the CPU to launch kernels one at a time. A profile of one
forward counted 1,562 `hipLaunchKernel` calls and 446 `hipExtModuleLaunchKernel`
calls, about 3,500 kernels on the GPU. A dependent chain of trivial kernels on
this GPU costs 2.9 µs each once the queue is full, and 8.3 µs for one launch
followed by a sync. Two thousand launches are a few milliseconds when the CPU
falls behind, which matches the 4–5 FPS the graph adds.

`reduce-overhead` captures the forward in one HIP graph. Steady state is one
`hipGraphLaunch` and one leftover `hipLaunchKernel` per pair. About 1,200
kernels still run; the CPU does not dispatch them individually. One graph
launch is under 10 µs.

## What a PCIe GPU would change

These are estimates from typical pinned `cudaMemcpy` throughput, not a
measurement on a discrete card. The APU column is measured.

| Exchange | Size | This APU | PCIe 5.0 x16 (~50 GB/s) | PCIe 4.0 x16 (~25 GB/s) | PCIe 4.0 x8 or 3.0 x16 (~12 GB/s) |
|---|---|---|---|---|---|
| New frame, host to device | 3.03 MB | 46 µs | ~61 µs (1.3×) | ~121 µs (2.6×) | ~253 µs (5.5×) |
| Flow, device to host | 2.02 MB | 34 µs | ~40 µs (1.2×) | ~81 µs (2.4×) | ~168 µs (5.0×) |

On PCIe 4.0 x16 the round trip is about 0.20 ms instead of 0.08 ms. The
forward would still read 26 ms. Weights move once (about 10 MB). The other
frame stays on the device because the two input buffers are swapped.

Mapped zero-copy is the exchange to avoid on a discrete GPU: the encoder would
read the frame over PCIe. This frame is 3 MB and fits in cache after one fill,
so that fill is again about 0.12 ms on PCIe 4.0 x16. It gets expensive only
when the GPU repeatedly re-reads a buffer larger than its cache. This model
does not.

A PCIe link does not change how many kernels the CPU dispatches. Each dispatch
is a doorbell of a few microseconds. The cost that shows up is the 2,000 eager
round trips, which this benchmark does not pay.

## Wall time, separate from the handoff

A full-video run once took 103–107 s (38 ms/pair reported, but 191 pairs at
that rate cannot fill 107 s). Two causes, neither of them the copy:

- PyTorch's default CPU pool used all 32 cores for `flow_to_image` and the
  2×2 composite. At 32 threads `flow_to_image` was 389 ms/frame and the
  composite 64 ms. At 4 threads they were 1.7 ms and 4.4 ms. The script now
  calls `torch.set_num_threads(4)` unless `OMP_NUM_THREADS` is set.
- The VAAPI probe encoded a 64×64 frame. This VCN encoder rejects anything
  below 128×128, so the script fell back to libx264 at 52 ms/frame. The probe
  is now 256×256, and the hardware encoder runs. After both fixes the same
  video was 7.2 s wall, 26.1 ms/pair.
