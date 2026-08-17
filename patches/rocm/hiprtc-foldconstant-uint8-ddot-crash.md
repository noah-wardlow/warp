# ROCm clang AMDGPU backend segfault: `FoldConstantArithmetic` on uint8 matrix ddot with `-DNDEBUG`

**Status:** to be filed upstream (LLVM / ROCm). Worked around in Warp's AMD port
(`warp/_src/build.py`, `WP_HIP_HIPRTC_MAX_SRC_BYTES`).

## Environment
- GPU / target: `gfx1151` (AMD Radeon 8060S, RDNA3.5)
- ROCm: 7.2.1
- Compiler: `AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.1 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)`
- Also crashes through HIPRTC (same clang backend), which takes down the host
  process because HIPRTC runs in-process.

## Summary
Compiling a HIP kernel that performs a **uint8 (i8) matrix double-dot** (`ddot`)
segfaults the clang AMDGPU backend **when `-DNDEBUG` is combined with `-O2` or
`-O3`**. The crash is in the instruction selector:

```
Running pass 'AMDGPU DAG->DAG Pattern Instruction Selection' on function
  '@check_mat_dot_uint8_..._cuda_kernel_forward'
 #4 llvm::SelectionDAG::FoldConstantArithmetic(...)
 #5 llvm::SelectionDAG::getNode(...)
 #9 llvm::SelectionDAG::LegalizeTypes()
#10 llvm::SelectionDAGISel::CodeGenAndEmitDAG()
...
clang++: error: unable to execute command: Segmentation fault (core dumped)
```

## Minimal reproducer
`repro_uint8_ddot.cu` (106 lines) — one uint8 2×2/4×4 matrix `ddot` kernel. It
includes Warp's `warp/native/builtin.h` (open source) for the `mat_t` / `ddot`
templates.

```sh
# CRASHES (segfault in FoldConstantArithmetic):
hipcc --genco --offload-arch=gfx1151 -O3 -DNDEBUG -std=c++17 \
      -I warp/native -I /opt/rocm/include repro_uint8_ddot.cu -o /dev/null

# COMPILES CLEANLY (any one of these):
hipcc --genco --offload-arch=gfx1151 -O3        -std=c++17 -I warp/native -I /opt/rocm/include repro_uint8_ddot.cu -o /dev/null  # drop -DNDEBUG
hipcc --genco --offload-arch=gfx1151 -O1 -DNDEBUG -std=c++17 -I warp/native -I /opt/rocm/include repro_uint8_ddot.cu -o /dev/null  # -O1
```

A fully self-contained (no `-I`) source for the LLVM team can be produced with:
```sh
hipcc -E -x hip -std=c++17 --offload-arch=gfx1151 -I warp/native -I /opt/rocm/include \
      repro_uint8_ddot.cu > repro.preprocessed.cu   # ~7 MB; still reproduces
```

## Isolation / bisection (each verified on the reproducer)
| Condition | Result |
|---|---|
| `-O3 -DNDEBUG` | **segfault** |
| `-O2 -DNDEBUG` | **segfault** |
| `-O1 -DNDEBUG` | ok |
| `-O3` (no `-DNDEBUG`) | ok |
| `-O3 -DNDEBUG` with `-real-true16` disabled or fp flags | still segfault (those flags are not the trigger) |
| same kernel but **float16** instead of uint8 | ok (uint8/small-int specific) |
| NVRTC (NVIDIA) and hipcc `-c` without `-DNDEBUG` | ok |

So the trigger is: **uint8 matrix `ddot` codegen + `-DNDEBUG` + `-O2/-O3`**. It is
content-specific, not size-specific (a single 106-line kernel reproduces). Warp
adds `-DNDEBUG` in release mode to strip device-side `assert()`s (≈3× smaller,
≈2× faster kernels); with the asserts removed, the i8 arithmetic feeds a constant
that `FoldConstantArithmetic` mishandles during type legalization.

## Warp-level reproducer (highest level)
Launching this kernel on a HIP device compiles it with `-DNDEBUG -O3` and
segfaults the process via HIPRTC:
```python
import warp as wp
u8mat22 = wp._src.types.matrix(shape=(2, 2), dtype=wp.uint8)
@wp.kernel
def k(v: wp.array(dtype=u8mat22), s: wp.array(dtype=u8mat22), out: wp.array(dtype=wp.uint8)):
    out[0] = wp.uint8(2) * wp.ddot(v[0], s[0])
```

## Workaround in Warp
`warp/_src/build.py` routes large HIP modules to out-of-process `hipcc --genco`
compiled **without `-DNDEBUG`**, so the segfault (a) cannot kill the Warp process
and (b) does not occur. Threshold `WP_HIP_HIPRTC_MAX_SRC_BYTES` (default 2 MB,
env-overridable; set 0 to force AOT for all HIP modules). Once the clang backend
bug is fixed upstream, this workaround can be removed.
