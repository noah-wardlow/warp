# Known issues: Warp on AMD ROCm/HIP (`gfx1151`)

This file describes the live `noah/gfx1151-rocm-port` branch only. Historical
results for older Warp, Newton, ROCm, and machine configurations remain in Git
history; they are not current compatibility claims.

Last audited: 2026-08-29.

## Current validation gate

The current Warp 1.18 branch builds and passes its compiler/diagnostic checks,
but the complete current-commit GPU suites have not finished.

With the GPU hidden, the current default suite completed 3,713 tests in two
workers: 3,166 passed, 547 skipped, and none failed or errored. This validates
CPU and backend-neutral behavior; it does not lift the GPU validation gate.

After an interrupted eight-worker Warp suite, the host ROCm 7.2.2 runtime
entered a state where `hipStreamCreate` blocks in a fresh process. The same
call blocks through `libamdhip64.so` without importing Warp, while `rocminfo`
and PyTorch default-stream allocation remain responsive. This is therefore a
host runtime/stream-state blocker, not evidence of a Warp initialization bug.

Check the raw operation before starting another suite:

```bash
timeout 15s python - <<'PY'
import ctypes

hip = ctypes.CDLL("libamdhip64.so")
stream = ctypes.c_void_p()
print("hipStreamCreate:start", flush=True)
status = hip.hipStreamCreate(ctypes.byref(stream))
print("hipStreamCreate:done", status, stream.value, flush=True)
PY
```

Expected output includes `hipStreamCreate:done` with status `0`. A timeout
requires a host reboot before graph, stream, or MuJoCo Warp results are valid.
Repeated test containers do not reset kernel/runtime state.

The current boot was not rebooted automatically because rebooting is an
administrative action and was not authorized.

## Platform capability gaps

These are explicit capabilities, not silent fallbacks.

### Conditional graph nodes

HIP/ROCm does not expose CUDA conditional graph nodes. Consequently:

- `warp.is_conditional_graph_supported()` returns `False` on HIP;
- `capture_if` and `capture_while` fail clearly when requested directly; and
- MuJoCo Warp sets `Model.opt.graph_conditional=False` while retaining normal
  capture and replay.

The solver runs its fixed iteration loop rather than graph-conditional early
exit. This can cost work after convergence but preserves correctness.

### Timing events inside captured graphs

ROCm 7.2 cannot record Warp timing events inside a captured graph. Ordinary
events and timing outside capture remain available.

`Device.supports_graph_event_timing` reports this. MuJoCo Warp's
`--event_trace` mode warns and traces eager calls; an untraced benchmark keeps
the graph-replay path.

### Texture mipmaps

The HIP texture adapter supports one native base level and explicit LOD
sampling. Native mipmapped arrays are unavailable, so
`Device.supports_texture_mipmaps` is `False` and six mipmap-specific texture
tests remain capability skips. Base textures, texture-handle arrays, and the
MuJoCo Warp renderer are supported.

### Conditional-only graph primitives and explicit allocation nodes

HIP lacks the CUDA conditional-node interface. The port also avoids depending
on captured HIP allocation/free node reconstruction: stable ordinary
allocations are associated with the graph lifetime instead. Tests that require
the exact CUDA allocation-node topology are not evidence for this HIP path.

### Unvalidated public surfaces

- Multi-GPU peer access, IPC, and cross-device graph behavior are not validated
  on the single-GPU Framework Desktop.
- Warp OpenGL registration is not validated on HIP. MuJoCo Warp's compute
  renderer does not require that registration path.
- HIP profiler start/stop is unsupported by the ROCm runtime and remains a
  capability skip.
- JAX, Paddle, and `usd-core` were absent from some test environments; their
  skips are not passes.
- Deterministic GPU-to-GPU and run-to-run atomic modes are not validated on
  HIP and remain scoped away from the default HIP suite.

## Numerical difference

One strict `float16` matrix inverse test produced `-31.375` on gfx1151 versus
the expected `-31.3125`: absolute difference `0.0625`, about `0.2%` and four
half-precision ULPs at that magnitude. The upstream tolerance is `0.05`.

The port does not widen that assertion. The broader aggregate ran 555 tests
with this one failure and six skips; a separate BF16 interop/tile set passed
8/8. Final classification requires rerunning the exact current commit after
the host reset.

## Memory behavior

### Async-pool reuse

ROCm 7.2's default opportunistic/internal-dependency reuse produced aliases to
still-live allocations during churn on gfx1151. Warp keeps the pool available
for graph capture but disables those unsafe reuse modes and uses ordinary
allocation outside capture.

`WARP_HIP_USE_ASYNC_POOL=1` bypasses this default and is diagnostic only.

### Stable capture allocations

Captured workloads in sparse, APIC, FEM, and MuJoCo Warp need stable addresses.
The HIP adapter pauses capture on its origin stream, makes an ordinary
allocation, resumes capture, and releases the address only after graph/user
ownership ends. Five targeted graph allocation regressions passed on the
preceding integrated port.

Forked-stream allocations retain the pooled capture path because ending an
origin capture from a forked stream is invalid. Multi-stream coverage must be
rerun on the final commit.

### UMA managed memory

The Radeon 8060S is a unified-memory architecture, but HIP managed allocation
is not the fastest default. An exploratory MuJoCo humanoid measurement at
1,024 worlds was about 18.0 ms/step with the hybrid allocator versus 3.7
ms/step with the normal allocation policy. `WARP_ENABLE_UMA_HYBRID=1` remains
an explicit readback-oriented option.

## Graph event ordering

On ROCm, a replayed graph's external event record is not reliably observed as
pending by an immediate waiter in the CUDA-equivalent way. Warp tracks the
producer stream and performs a conservative synchronization before a dependent
external wait. This is stronger ordering than CUDA but localizes correctness at
the graph/event seam.

The integrated stream regression ran 25 tests: 22 passed and three skipped for
hardware/capability requirements. It must be repeated on the current commit
after reboot.

## Compiler behavior

### RDNA 3.5 true16 register mode

ROCm 7.2's AMDGPU backend can misallocate `_hi16`/`_lo16` subregisters in
packed FP16/BF16 code at optimization levels used by Warp. The build disables
the `real-true16` target feature for HIP device compilation.

A combined HIP host/device compile prints:

```text
'-real-true16' is not a recognized feature for this target (ignoring feature)
```

Driver tracing proves the gfx1151 device job receives and accepts
`-real-true16`; a device-only compile is warning-free. The message comes from
the x86 host job, which correctly ignores the AMDGPU-only feature. ROCm 7.2.2
cannot wrap the multi-argument `-Xclang` pair with `-Xarch_device`, so the
benign host diagnostic remains.

### HIPRTC precompiled headers and source size

HIPRTC does not support Warp's CUDA precompiled-header path. The port disables
that request. Large generated sources use `hipcc --genco` above the measured
`WARP_HIP_HIPRTC_MAX_SRC_BYTES` threshold to avoid HIPRTC compiler failures;
set the threshold to `0` only when diagnosing the AOT path.

## cuBQL and rendering

The current branch ports the bundled cuBQL builder and traversal templates to
HIP/hipCUB-compatible primitives. Completed evidence on the integrated port:

- 74 runnable geometry tests passed; 18 `usd-core` cases skipped;
- 20 fresh processes each passed the three-test BVH query regression;
- 54 MuJoCo Warp renderer/BVH tests passed; four optional-dependency cases
  skipped; and
- one-million-AABB cuBQL construction measured 0.201-0.216 s versus
  0.465-0.491 s for CPU SAH.

LBVH remains the fastest builder in isolation, but the measured 1,024-world
renderer reached 111,988 world-frames/s with the cuBQL/default path versus
76,256 with LBVH. MuJoCo Warp therefore chooses cuBQL when
`warp.is_cubql_available()` is true and SAH otherwise.

These results require final-commit repetition after the host runtime reset.

## Reporting a new failure

Record enough state to distinguish a port defect from a host runtime failure:

```bash
uname -a
cat /proc/sys/kernel/random/boot_id
rocminfo | grep -E 'Name:|Marketing Name:|Compute Unit:|Wavefront Size:'
hipconfig --full
git rev-parse HEAD
```

Also record the exact container/image or native ROCm tree, `WARP_CACHE_PATH`,
test command, exit status, and complete output. Give different native builds
different cache directories; Warp module hashes do not represent every change
inside `warp.so`.
