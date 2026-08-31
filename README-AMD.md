# Warp on AMD ROCm/HIP

This branch ports Warp 1.18 to AMD HIP/ROCm and tracks NVIDIA Warp `main`.
The implementation is developed and measured on the Radeon 8060S
(`gfx1151`) in a Framework Desktop.

> **Validation status:** implementation commit
> `58146889520006b5e56cad1b08f1c4fe22bb2eea` has a clean native gfx1151
> build and a complete 7,354-test GPU pass. Matching MuJoCo Warp implementation
> commit `ab9d3d121ae8d8d6de3b7efd0d2de563cea6eeb9` also passes its complete
> GPU suite against that binary. Later documentation-only commits do not change
> the tested implementation. See [KNOWN_ISSUES-AMD.md](KNOWN_ISSUES-AMD.md)
> for explicit capability gaps and unvalidated surfaces.

## Supported target

| Item | Validated configuration |
| --- | --- |
| GPU | AMD Radeon 8060S, `gfx1151`, 40 compute units, wave32 |
| Host | Framework Desktop, Ubuntu 26.04, kernel 7.0.0-30 |
| Runtime | ROCm 7.2.2 (`HIP 7.2.53211`) |
| Container | `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` |
| Python | 3.12 |

Other AMD architectures are not validated by this branch. `build_lib.py`
accepts other HIP architecture strings, but a successful compile is not a
correctness or performance claim.

Confirm the target before building:

```bash
rocminfo | grep -E 'Name:|Marketing Name:|Compute Unit:|Wavefront Size:'
```

## Build

Install Git LFS before cloning so test and example assets are materialized.
Use an isolated Python environment and an installed ROCm tree:

```bash
git lfs install
git clone --branch noah/gfx1151-rocm-port \
  https://github.com/noah-wardlow/warp.git warp-rocm
cd warp-rocm

uv sync --extra dev
uv run ./build_amd.sh
uv pip install --no-deps --editable .
uv run python tools/run_gfx1151_smoke.py
```

`build_amd.sh` resolves these inputs once and forwards additional arguments to
`build_lib.py`:

- `ROCM_PATH`: ROCm installation, default `/opt/rocm`.
- `HIP_ARCH`: comma-separated architecture list, default `gfx1151`.
- `PYTHON`: interpreter used by the build script; normally inherited from the
  active environment.

The equivalent explicit build is:

```bash
uv run build_lib.py \
  --no-cuda \
  --rocm-path=/opt/rocm \
  --hip-arch=gfx1151 \
  --quick
```

### Isolated container build

The measured configuration uses the ROCm 7.2.2 PyTorch image. Run the
container as the owner of the mounted checkout so generated files remain
user-owned:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --device /dev/kfd \
  --device /dev/dri \
  -v "$PWD:/warp" \
  -w /warp \
  rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  bash

python build_lib.py \
  --no-cuda \
  --rocm-path=/opt/rocm \
  --hip-arch=gfx1151 \
  --quick \
  --jobs=4
python tools/run_gfx1151_smoke.py
```

## Backend design

The public device vocabulary remains compatible with Warp: GPU aliases are
still `cuda:N`, while `Device.is_hip` and `Device.arch_str` distinguish the
backend. Platform behavior is concentrated behind the `Device` interface and
the native HIP adapter rather than scattered through callers.

### Build and compiler seam

- `build_lib.py` and `build_amd.sh` select HIP, the ROCm tree, and one or more
  `gfx*` targets.
- HIP native translation is localized in `warp/native/hip_util.h` and the HIP
  branches of `cuda_util.cpp` and `warp.cu`.
- JIT compilation uses HIPRTC for ordinary modules and `hipcc --genco` for
  sources over the measured HIPRTC size threshold.
- CUDA-only mathDx and precompiled-header behavior are disabled explicitly.

### Device capability seam

Callers should query capabilities instead of comparing CUDA compute
capabilities on HIP:

- `Device.supports_graph_capture`
- `Device.supports_graph_event_timing`
- `Device.supports_cubql`
- `Device.supports_float16`
- `Device.supports_bfloat16`
- `Device.supports_texture_mipmaps`
- `warp.is_conditional_graph_supported()`

This interface has two real adapters: CUDA and HIP. CPU behavior remains the
third execution path where the existing Warp interface already defines it.

### Memory and graph seam

ROCm 7.2's asynchronous pool can recycle memory before a stream-ordered free
is safe under allocation churn. The HIP adapter therefore:

1. disables unsafe opportunistic and internal-dependency pool reuse;
2. uses ordinary device allocation outside graph capture;
3. pauses an origin-stream capture, performs a stable ordinary allocation,
   then resumes capture; and
4. retains those stable addresses until both the graph and user reference
   release them.

This keeps native graph replay enabled without relying on captured HIP memory
allocation nodes. Cross-graph external-event dependencies use a conservative
producer-stream synchronization because ROCm does not expose CUDA-identical
event state during replay.

### Textures, geometry, and rendering

- HIP texture objects expose one real base level and explicit LOD sampling.
  Native mipmaps are reported as unavailable instead of emulated.
- The bundled cuBQL GPU BVH builder and traversal code compile through HIP and
  hipCUB-compatible primitives.
- `warp.is_cubql_available()` reports whether the native library contains that
  implementation. MuJoCo Warp selects cuBQL when present and SAH otherwise.

### Low precision and atomics

`float16` and `bfloat16` selection uses device capabilities rather than CUDA
SM numbers. The ROCm 7.2 compiler's problematic RDNA 3.5 true16 register mode
is disabled for device compilation. A combined HIP compile prints one benign
host-side warning because the x86 compiler ignores that AMDGPU-only feature;
the gfx1151 device compilation accepts it.

HIP device-side half conversion uses an explicit `v_cvt_f16_f32` /
`v_cvt_f32_f16` barrier. Without that barrier, the optimizing compiler can
fold adjacent float-to-half and half-to-float operations and silently remove
the half-precision rounding point. The regression checks the rounded result;
the upstream matrix-inverse tolerance remains unchanged.

## Measured validation

All counts below describe completed commands. Skips are retained as skips.

| Surface | Result | Scope |
| --- | ---: | --- |
| Warp 1.18 clean native build | pass | Exact `58146889`; release HIP runtime, HIPRTC, LLVM, cuBQL, `gfx1151` |
| Warp default GPU suite | 7,104 passed, 250 skipped | 7,354 tests, serial, 0 failures/errors in 8,744.773 s |
| Exact-binary focused gates | pass | gfx1151 smoke; 1 FP16 inverse; 8 FP16; 5 tile diagnostics; 1,213 async |
| Async fresh-process reliability | 7 × 1,213 passed | Six pre-commit audit runs plus one exact rebuilt-binary run; no quarantine remains |
| Current default suite, GPU hidden | 3,166 passed, 547 skipped | 3,713 tests, 0 failures/errors; CPU and availability surfaces |
| MuJoCo Warp full GPU suite | 1,324 passed, 37 skipped | 1,361 cases plus 8 passed subtests, serial, 0 failures/errors in 560.26 s |
| MuJoCo Warp warning contract | 7 passed | Two backend-notice variants and five MULTICCD warning cases |

Representative completed performance measurements on the same Radeon 8060S:

- MuJoCo humanoid, 8,192 worlds × 100 steps: median **451,360
  world-steps/s** across five fresh processes (450,416–452,443); all worlds
  converged.
- Batched primitives renderer, 1,024 worlds at 64×64 RGB+depth: median
  **91,646 world-frames/s** across five fresh processes (91,486–92,009); all
  worlds converged.
- Random 1,000,000-AABB build: median **0.2084 s** with HIP cuBQL versus
  **0.4780 s** with CPU SAH. LBVH built in **0.01385 s** but produced lower
  end-to-end renderer throughput in the earlier controlled builder comparison.

These measurements characterize this port; they are not a cross-vendor
comparison. The humanoid and renderer figures above were repeated on exact
commits `58146889` and `ab9d3d1`; the standalone builder comparison predates
those final implementation commits and is retained as design evidence.

## Validation commands

Run narrow checks before the full suite:

```bash
uv run python tools/run_gfx1151_smoke.py
uv run python -m unittest \
  warp.tests.test_compilation.TestCompilation.test_hip_merged_include_resolution \
  warp.tests.test_diagnostics
```

Run the gfx1151 suite serially. Multiple Python workers can contend in the
ROCm/HSA runtime on this integrated GPU and obscure whether a failure belongs
to Warp or to process-level device contention:

```bash
uv run --extra dev python -m warp.tests \
  -s default \
  --serial-fallback \
  --junit-report-xml warp-rocm-gfx1151.xml
```

An interrupted or timed-out suite is inconclusive. Before rerunning after a
runtime stall, verify a raw HIP stream can be created in a fresh process; see
[KNOWN_ISSUES-AMD.md](KNOWN_ISSUES-AMD.md).

## MuJoCo Warp

Use the matching source branch; do not let environment synchronization replace
the HIP Warp checkout with the published CUDA wheel:

```bash
git clone --branch noah/gfx1151-rocm-port \
  https://github.com/noah-wardlow/mujoco_warp.git mujoco_warp-rocm
cd mujoco_warp-rocm

uv sync --extra dev
uv pip install --no-deps --editable ../warp-rocm
uv run --no-sync pytest
```

MuJoCo Warp automatically disables only unavailable conditional graph nodes.
Ordinary graph replay remains enabled. Its `--event_trace` mode uses eager
execution on HIP because timing events cannot be embedded in a captured graph;
run without `--event_trace` for graph-replay throughput.

## Measured overrides

Defaults are the selected configuration. These overrides exist for diagnosis
or a measured workload, not as additional supported modes:

| Setting | Effect |
| --- | --- |
| `WARP_HIP_FAST_FP_ATOMICS=1` | Select native FP32 atomics for contention patterns where they were measured faster. Also available as the `hip_fast_fp_atomics` module option. |
| `WARP_HIP_USE_ASYNC_POOL=1` | Use the HIP async pool for ordinary allocations. This bypasses the safer gfx1151 default. |
| `WARP_HIP_STABLE_CAPTURE_ALLOCS=0` | Disable stable capture allocations and retain HIP allocation nodes. |
| `WARP_HIP_GRAPH_FREE_NODES=1` | Enable HIP graph free-node insertion. This is not the validated default. |
| `WARP_ENABLE_UMA_HYBRID=1` | Use managed memory for allocations of at least 64 KiB on a UMA device; measured substantially slower for batched MuJoCo physics. |
| `WARP_HIP_HIPRTC_MAX_SRC_BYTES=0` | Route all JIT modules through the AOT `hipcc --genco` path. |

Do not combine allocator overrides unless the combination is the subject of a
new correctness and performance experiment.
