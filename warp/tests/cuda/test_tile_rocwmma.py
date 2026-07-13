# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
# AMD rocWMMA tile_matmul unit test
# File: warp/tests/cuda/test_tile_rocwmma.py

"""
Unit tests for AMD rocWMMA tile_matmul fast path.
Uses plain pytest. Tests skip if no HIP device available.
"""

import pytest
import numpy as np
import warp as wp


def get_hip_device():
    """Return first HIP device string, or None."""
    wp.init()
    for d in wp.get_cuda_devices():
        if d.is_hip:
            return str(d)
    return None


HIP_DEVICE = get_hip_device()
pytestmark = pytest.mark.skipif(HIP_DEVICE is None, reason="No HIP device")


# Kernels: all take inputs by reference, C is in-place output
@wp.kernel
def tile_matmul_16x16_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C: wp.array2d(dtype=float),
):
    """16x16 tile matmul: C = A @ B (overwrite C)"""
    a = wp.tile_load(A, shape=(16, 16), storage="shared")
    b = wp.tile_load(B, shape=(16, 16), storage="shared")
    c = wp.tile_zeros(dtype=float, shape=(16, 16), storage="shared")
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel
def tile_matmul_16x16_k32_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C: wp.array2d(dtype=float),
):
    """16x16 output, K=32"""
    a = wp.tile_load(A, shape=(16, 32), storage="shared")
    b = wp.tile_load(B, shape=(32, 16), storage="shared")
    c = wp.tile_zeros(dtype=float, shape=(16, 16), storage="shared")
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


@wp.kernel
def tile_matmul_alpha_beta_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
    C: wp.array2d(dtype=float),
    alpha: float,
    beta: float,
):
    """C = alpha * A @ B + beta * C (accumulate in-place)"""
    a = wp.tile_load(A, shape=(16, 16), storage="shared")
    b = wp.tile_load(B, shape=(16, 16), storage="shared")
    c = wp.tile_load(C, shape=(16, 16), storage="shared")
    wp.tile_matmul(a, b, c, alpha=alpha, beta=beta)
    wp.tile_store(C, c)


def _matmul(M, N, K, alpha=1.0, beta=0.0):
    """Run tile_matmul and return (result_np, numpy_reference)."""
    device = HIP_DEVICE
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((M, K)).astype(np.float32)
    B_np = rng.standard_normal((K, N)).astype(np.float32)
    C_np = rng.standard_normal((M, N)).astype(np.float32) if beta != 0.0 \
           else np.zeros((M, N), dtype=np.float32)
    ref = alpha * (A_np @ B_np) + beta * C_np

    A_wp = wp.array(A_np, dtype=float, device=device)
    B_wp = wp.array(B_np, dtype=float, device=device)
    C_wp = wp.array(C_np.copy(), dtype=float, device=device)

    if alpha == 1.0 and beta == 0.0 and K == 16:
        wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                       inputs=[A_wp, B_wp, C_wp],
                       block_dim=64, device=device)
    elif alpha == 1.0 and beta == 0.0 and K == 32:
        wp.launch_tiled(tile_matmul_16x16_k32_kernel, dim=(1,),
                       inputs=[A_wp, B_wp, C_wp],
                       block_dim=64, device=device)
    else:
        wp.launch_tiled(tile_matmul_alpha_beta_kernel, dim=(1,),
                       inputs=[A_wp, B_wp, C_wp, float(alpha), float(beta)],
                       block_dim=64, device=device)
    return C_wp.numpy(), ref


def test_tile_matmul_16x16_identity():
    """B=identity → result equals A."""
    device = HIP_DEVICE
    A_np = np.random.randn(16, 16).astype(np.float32)
    B_np = np.eye(16, dtype=np.float32)
    A_wp = wp.array(A_np, dtype=float, device=device)
    B_wp = wp.array(B_np, dtype=float, device=device)
    C_wp = wp.zeros((16, 16), dtype=float, device=device)
    wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                   inputs=[A_wp, B_wp, C_wp],
                   block_dim=64, device=device)
    np.testing.assert_allclose(C_wp.numpy(), A_np, atol=1e-4)


def test_tile_matmul_16x16_zeros():
    """B=0 → result is zero."""
    device = HIP_DEVICE
    A_wp = wp.ones((16, 16), dtype=float, device=device)
    B_wp = wp.zeros((16, 16), dtype=float, device=device)
    C_wp = wp.zeros((16, 16), dtype=float, device=device)
    wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                   inputs=[A_wp, B_wp, C_wp],
                   block_dim=64, device=device)
    np.testing.assert_allclose(C_wp.numpy(), np.zeros((16, 16)), atol=1e-6)


def test_tile_matmul_16x16_random():
    result, ref = _matmul(16, 16, 16)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)


def test_tile_matmul_16x16_k32():
    result, ref = _matmul(16, 16, 32)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)


def test_tile_matmul_alpha_beta():
    """alpha=2.0, beta=0.5"""
    result, ref = _matmul(16, 16, 16, alpha=2.0, beta=0.5)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)


def test_tile_matmul_beta_one():
    """alpha=1, beta=1: accumulate"""
    result, ref = _matmul(16, 16, 16, alpha=1.0, beta=1.0)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4)


def test_tile_matmul_accumulate_false_ignores_beta():
    """Overwrite mode: initial C=999 must be ignored.
    Bug fixed: Accumulate template param was ignored in rocWMMA path."""
    device = HIP_DEVICE
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((16, 16)).astype(np.float32)
    B_np = rng.standard_normal((16, 16)).astype(np.float32)
    C_np = np.ones((16, 16), dtype=np.float32) * 999.0
    expected = A_np @ B_np

    A_wp = wp.array(A_np, dtype=float, device=device)
    B_wp = wp.array(B_np, dtype=float, device=device)
    C_wp = wp.array(C_np.copy(), dtype=float, device=device)
    # tile_matmul_16x16_kernel uses overwrite (Accumulate=false), ignores C contents
    wp.launch_tiled(tile_matmul_16x16_kernel, dim=(1,),
                   inputs=[A_wp, B_wp, C_wp],
                   block_dim=64, device=device)
    np.testing.assert_allclose(C_wp.numpy(), expected, atol=1e-4, rtol=1e-4,
        err_msg="Accumulate=false: initial C=999 leaked into result")


def test_tile_matmul_alpha2_beta1():
    """alpha=2, beta=1: catches alpha-applied-to-wrong-operand bug.
    Bug: old code gave alpha*(A@B + beta*C) = 2*(A@B+C); correct: 2*A@B+C."""
    result, ref = _matmul(16, 16, 16, alpha=2.0, beta=1.0)
    np.testing.assert_allclose(result, ref, atol=1e-4, rtol=1e-4,
        err_msg="alpha=2,beta=1: bug gives 2*(A@B+C) instead of 2*A@B+C")


def test_tile_matmul_alpha0_beta1():
    """alpha=0, beta=1: result must be C_in exactly.
    Bug: old code gave 0*(A@B+C)=0; correct: 0*A@B + C = C."""
    device = HIP_DEVICE
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((16, 16)).astype(np.float32)
    B_np = rng.standard_normal((16, 16)).astype(np.float32)
    C_np = rng.standard_normal((16, 16)).astype(np.float32)
    expected = C_np.copy()

    A_wp = wp.array(A_np, dtype=float, device=device)
    B_wp = wp.array(B_np, dtype=float, device=device)
    C_wp = wp.array(C_np.copy(), dtype=float, device=device)
    wp.launch_tiled(tile_matmul_alpha_beta_kernel, dim=(1,),
                   inputs=[A_wp, B_wp, C_wp, float(0.0), float(1.0)],
                   block_dim=64, device=device)
    np.testing.assert_allclose(C_wp.numpy(), expected, atol=1e-5,
        err_msg="alpha=0,beta=1: result should be C_in, not 0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
