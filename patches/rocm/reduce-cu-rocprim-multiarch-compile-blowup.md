# ROCm hipCUB/rocPRIM: `reduce.cu` compile-time blowup with a fused custom iterator

**Status:** worked around in Warp's AMD port (`warp/native/reduce.cu`, one
`noinline` on the fused inner-product iterator). Distinct from the uint8-ddot
`FoldConstantArithmetic` crash in `hiprtc-foldconstant-uint8-ddot-crash.md`.

## Environment
- GPU / target: `gfx1151` (AMD Radeon 8060S, RDNA3.5), ROCm 7.2.1
- Compiler: AMD clang 22.0.0git (roc-7.2.1)

## Summary
Compiling `warp/native/reduce.cu` intermittently spins the AMDGPU backend at
99 % CPU with **flat ~0.6 GB RSS** (not a memory blowup) and effectively never
finishes — observed hanging >70 min in a real build, and reproducibly >600 s in
isolation. It is **not** flag-sensitive: `-O3 -DNDEBUG`, `-O3` (no `-DNDEBUG`),
and `-O1 -DNDEBUG` all hang, so the report's uint8-ddot mitigations do not apply.

## Root cause
`array_inner_device` reduces with `cub::DeviceReduce::Sum` over a **custom fused
iterator** (`cub_inner_product_iterator`) whose `operator*` runs a per-element
loop calling `dot`. hipCUB's `DeviceReduce` uses rocPRIM `default_config`, which
instantiates the entire reduce pipeline (`trampoline_kernel`, `reduce_impl`,
`warp_reduce_dpp`, `warp_reduce_shuffle`) **once per known GPU arch**
(`target_arch` 1201/1200/1102/1100/… — ~a dozen). The fused iterator's loop is
inlined into *every* per-arch kernel, so the backend must optimize a dozen large
kernels. That is what blows up. `array_sum_device` in the same file reduces with
a **trivial** iterator (`operator*` returns `*ptr`) and compiles fine under the
identical multi-arch instantiation — confirming the cost is the fused
dereference, not the multi-arch dispatch itself.

Pinpointed with `-mllvm -print-before-all`: the last passes before the hang are
on `rocprim::…trampoline_kernel<…, target_arch::N, …cub_inner_product_iterator…>`
for successive `N`.

## Fix (in-tree)
Mark the iterator's `compute_value` **`__attribute__((noinline))` on HIP** so it
is emitted once and *called* per element instead of inlined into each per-arch
kernel. Compile time drops from “never” to **~50 s**, deterministically (3/3
parallel builds). Guarded by `#if defined(__HIP_PLATFORM_AMD__)`, so CUDA/NVRTC
codegen is unchanged. Runtime impact is negligible: array inner-product/L2 is
memory-bound and not a Warp hot path. Verified: `test_array_reduce` 16/16,
`test_utils` 78/78.

Once rocPRIM/hipCUB offers single-arch config selection (or stops inlining
fused iterators across all archs), the attribute can be removed.
