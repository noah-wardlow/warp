[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Warp — AMD ROCm port (v1.13.0 base)

This repository is AMD's port of [NVIDIA Warp](https://github.com/NVIDIA/warp) to ROCm/HIP.
It tracks upstream Warp at tag **`v1.13.0`** and adds a HIP backend that targets AMD CDNA3 GPUs.

For upstream Warp's documentation, language reference, and changelog see
**[Documentation](https://nvidia.github.io/warp/)** and
[CHANGELOG.md](https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md).

---

## What this fork is and isn't

Warp is a Python framework that JIT-compiles regular Python kernels to efficient GPU code.
Upstream targets NVIDIA GPUs via CUDA/NVRTC. **This fork lets the same Python kernels run on
AMD Instinct GPUs via HIP/HIPRTC**, while keeping the upstream CUDA path bit-for-bit
identical when you build with `--no-rocm-path`.

It is *not* a full ROCm replacement for every Warp feature — the
[Deferred features](#deferred-features-on-the-hip-path) section enumerates what's currently
gated off on HIP. Everything else — kernels, tile primitives, BVH, mesh queries, HashGrid,
volumes, autograd, DLPack/PyTorch/JAX interop, fp16, bf16, and the example suite — is in scope.

## Supported configurations

| Component       | Supported                                              | Notes |
|-----------------|--------------------------------------------------------|-------|
| GPU             | `gfx942` (MI300x, MI325x); `gfx950` (MI355x)           | `gfx90a` is not in the default target list |
| ROCm            | 7.x; tested with TheRock 7.9 nightly snapshots         | See `docker/rocm_ci/Dockerfile.TheRock_ci` |
| Host OS         | Linux (Ubuntu 24.04 in CI; modern glibc required)      | Windows HIP is rejected at build time |
| Python          | 3.10+                                                  | Same minimum as upstream v1.13.0 |
| PyTorch         | 2.9+ ROCm wheels (`pip --pre torch` from rocm.nightlies)| Used by `example_torch_diffray.py`; optional |

NVIDIA hardware is supported unchanged: build without `--rocm-path` (and with `ROCM_PATH`
unset / no `hipcc` on `PATH`) and the result is identical to upstream `v1.13.0`.

## Quick Start

```python
import warp as wp
import numpy as np

num_particles = 1_000_000
dt = 0.01

@wp.kernel
def gravity_step(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    position = pos[i]
    dist_sq = wp.length_sq(position) + 0.01
    acc = -1000.0 / dist_sq * wp.normalize(position)
    vel[i] = vel[i] + acc * dt
    pos[i] = pos[i] + vel[i] * dt

rng = np.random.default_rng(42)
positions = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3)
velocities = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3)

for _ in range(100):
    wp.launch(gravity_step, dim=num_particles, inputs=[positions, velocities])

print(positions.numpy())
```

## Building from source

### Prerequisites

* A ROCm 7.x install (`hipcc` reachable via `PATH`, `ROCM_PATH`, `ROCM_HOME`, or under `/opt/rocm`)
* GCC 14 / libstdc++ 14 (Ubuntu 24.04 default works out of the box)
* Python 3.10+ with `numpy` available for the build script

### Build

```bash
git clone https://github.com/ROCm/warp.git
cd warp

# Auto-detects ROCm; defaults --hip-arch to gfx942,gfx950
python build_lib.py --jobs $(nproc)

pip install -e .
```

To target a single GPU family or a non-default arch list:

```bash
python build_lib.py --rocm-path /opt/rocm --hip-arch gfx942
# or
python build_lib.py --rocm-path /opt/rocm --hip-arch gfx950
# or via env vars
HIP_ARCH=gfx942,gfx950 python build_lib.py
```

To build the CUDA path on an NVIDIA system, run as upstream does (no `--rocm-path`,
no ROCm autodetected):

```bash
python build_lib.py --jobs $(nproc) --cuda-path /usr/local/cuda
```

To build a CPU-only library (no CUDA, no HIP):

```bash
python build_lib.py --no-cuda
```

### Build flags reference

| Flag                                  | Default                | Effect |
|---------------------------------------|------------------------|--------|
| `--rocm-path PATH`                    | autodetect             | Enables HIP build using this ROCm SDK |
| `--hip-arch gfx942[,gfx950,...]`      | `gfx942,gfx950`        | AMD GPU targets passed to `hipcc --offload-arch` |
| `--cuda-path PATH`                    | autodetect             | Path to CUDA Toolkit (mutually exclusive with `--no-cuda`) |
| `--no-cuda`                           | off                    | Skip CUDA path entirely (CPU-only or HIP-only) |
| `--no-use-libmathdx`                  | off                    | Disable cuBLASDx/cuFFTDx/cuSOLVERDx (auto-disabled on HIP) |
| `--quick`                             | off                    | Compile for fewer GPU archs to speed up dev builds |

The HIP path implicitly disables libmathdx (`mathdx_enabled=0`) since cuBLASDx/cuFFTDx/cuSOLVERDx
have no AMD equivalent. Tile FFT and tile matmul that depend on libmathdx will raise
`NotImplementedError` on HIP devices.

### Container build

A self-contained TheRock-based ROCm image is provided at
`docker/rocm_ci/Dockerfile.TheRock_ci`. Default arch is `gfx942`; override via build args
to target other families:

```bash
docker buildx build \
    --build-context warp_src=. \
    -f docker/rocm_ci/Dockerfile.TheRock_ci \
    --target warp_base \
    --build-arg ROCM_AMDGPU_TARGETS=gfx942 \
    --build-arg PYTORCH_ROCM_ARCH=gfx942 \
    -t warp:rocm-gfx942 .
```

## Deferred features on the HIP path

The following features are present on the CUDA path but currently gated off when running on
AMD GPUs. Their absence is reported via `NotImplementedError` (Python) or by skipping the
corresponding tests:

| Feature                          | Status on HIP                   | Reason |
|----------------------------------|---------------------------------|--------|
| CUDA Graph capture               | `wp.capture_begin()` returns False; tests skipped | HIP graph capture has correctness gaps in ROCm 7.x |
| Conditional graph nodes          | Not supported                   | No HIP equivalent for `cuGraphAddNode(CONDITIONAL)` |
| APIC GPU-side capture            | Not supported                   | Depends on cuMemPool* APIs without ROCm equivalents |
| cuBQL BVH backend                | Disabled; falls back to Karras LBVH | cuBQL templates assume NVIDIA-only intrinsics |
| RMM allocator                    | Disabled                        | RMM has no HIP backend |
| GLTextureResource                | Disabled                        | OpenGL/HIP interop is not wired up |
| Texture sampling (1D/2D/3D)      | Returns zero                    | HIP image support is off on CDNA3 |
| `libmathdx` (cuBLASDx/cuFFTDx/cuSOLVERDx) | Disabled               | NVIDIA-only library |
| `tile_matmul`, `tile_cholesky`, `tile_diag_solve`, `tile_lu_solve`, `tile_qr_solve` | `NotImplementedError` | Bind through cuBLASDx / cuSOLVERDx (libmathdx); deferred until a rocBLAS/rocSOLVER tile path lands |
| `tile_fft`                       | `NotImplementedError`           | Binds through cuFFTDx (libmathdx); needs a rocFFT tile path |

The remaining tile primitives — `tile_load`, `tile_store`, `tile_view`, `tile_arange`,
`tile_broadcast`, `tile_reduce`, `tile_scan`, `tile_sort`, `tile_atomic_add`, and the
`tile_bvh` / `tile_mesh` queries — *are* supported on HIP. The HIP build widens
`tile_mask_t` to 64 bits to match the AMD warp size, swaps `WP_TILE_SHARED_ARRAY` to a
raw-byte representation (works around HIPRTC's refusal of `__shared__` arrays of
non-trivial types), and applies `__attribute__((optnone))` on `tile_atomic_add` to
defeat a HIPRTC-clang miscompile.

If you need any of these, please file an issue against this fork — most are *deferred*, not
*rejected*, and will follow once the core port is stable on `v1.13.0`.

## Architectural notes for contributors

The HIP backend is structured to keep upstream code paths intact:

* `warp/native/hip_util.h` — pure C++ shim that maps `cu*` / `nvrtc*` symbols to the matching
  HIP entry points. CUDA-only files include `hip_util.h` under `#if defined(__HIP_PLATFORM_AMD__)`
  and use the unmodified CUDA symbol names everywhere else.
* `warp/native/cuda_util.cpp` and `warp.cu` — `pfn_cu*_f` function pointers are gated; on HIP
  they resolve via `hipGetProcAddress`. Image, conditional-graph, and memory-batch entry points
  return `HIP_ERROR_NOT_SUPPORTED` (or compile-out) on HIP.
* `warp/native/tile.h` — warp size is `64`, `tile_mask_t` widens to `uint64`, `WP_TILE_SHARED_ARRAY`
  uses raw byte buffers to defeat HIPRTC's refusal of `__shared__` non-trivial types.
* `warp/native/bvh.cu` — Karras LBVH builder replaces the persistent-scratch builder on HIP only;
  the CUDA path is preserved verbatim under `#if !defined(__HIP_PLATFORM_AMD__)`.
* `warp/native/mesh.h` — HIP gets a stack-based speculative ray traversal because HIP atomics
  / fences don't satisfy the original implementation's invariants.
* `warp/_src/context.py` — `Device.is_hip`, `Device.arch_str`, `Runtime.is_hip` plus arch-string
  return paths from `get_cuda_supported_archs()`.

The upstream Warp test suite is run with HIP-aware skips for the deferred features above.

## License

Apache 2.0, identical to upstream Warp. See [LICENSE.md](./LICENSE.md).
