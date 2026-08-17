# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def _fill_scalar(a: wp.array(dtype=wp.float32), base: float):
    i = wp.tid()
    a[i] = base + wp.float32(i)


@wp.kernel
def _fill_vec(a: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    a[i] = wp.vec3(wp.float32(i), 1.0, 2.0)


def test_readback_buffer_scalar(test, device):
    n = 4096
    buf = wp.ReadbackBuffer(shape=n, dtype=wp.float32, device=device)
    wp.launch(_fill_scalar, dim=n, inputs=[buf.array, 10.0], device=device)
    got = buf.numpy()
    test.assertEqual(got.shape, (n,))
    np.testing.assert_allclose(got, 10.0 + np.arange(n, dtype=np.float32))
    buf.release()


def test_readback_buffer_vec(test, device):
    n = 1000
    buf = wp.ReadbackBuffer(shape=(n,), dtype=wp.vec3, device=device)
    wp.launch(_fill_vec, dim=n, inputs=[buf.array], device=device)
    got = buf.numpy()
    test.assertEqual(got.shape, (n, 3))
    expected = np.stack([np.arange(n), np.ones(n), 2.0 * np.ones(n)], axis=1)
    np.testing.assert_allclose(got, expected)
    buf.release()


def test_numpy_view_device_local_raises(test, device):
    # A device-local (non-managed) array has no CPU-accessible view and must
    # refuse numpy_view(). Skip when a unified-memory hybrid allocator is active.
    d = wp.get_device(device)
    if not d.is_cuda:
        return
    a = wp.zeros(64, dtype=wp.float32, device=device)
    is_managed = getattr(d.current_allocator, "is_managed", None)
    if is_managed is not None and is_managed(a.ptr):
        return
    with test.assertRaises(RuntimeError):
        a.numpy_view()


# module="unique" because hip_fast_fp_atomics is a module-level option
@wp.kernel(module_options={"hip_fast_fp_atomics": True}, module="unique")
def _fast_atomic_add(dst: wp.array(dtype=wp.float32)):
    wp.tid()
    wp.atomic_add(dst, 0, 1.0)


def test_readback_buffer_survives_fast_fp_atomics(test, device):
    # A ReadbackBuffer is managed memory, and HIP's hardware float atomic
    # discards every update to host-coherent memory. The buffer is marked
    # coarse-grained at allocation so the atomic lands; without that this sums
    # to zero with no error raised.
    if not getattr(device, "is_uma", False):
        test.skipTest("requires a unified-memory device")

    n = 1 << 20
    buf = wp.ReadbackBuffer(shape=(1,), dtype=wp.float32, device=device)
    wp.launch(_fast_atomic_add, dim=n, inputs=[buf.array], device=device)
    wp.synchronize_device(device)
    test.assertEqual(float(buf.numpy()[0]), float(n))


devices = get_test_devices()


class TestUmaReadback(unittest.TestCase):
    pass


add_function_test(
    TestUmaReadback,
    "test_readback_buffer_survives_fast_fp_atomics",
    test_readback_buffer_survives_fast_fp_atomics,
    devices=devices,
)
add_function_test(TestUmaReadback, "test_readback_buffer_scalar", test_readback_buffer_scalar, devices=devices)
add_function_test(TestUmaReadback, "test_readback_buffer_vec", test_readback_buffer_vec, devices=devices)
add_function_test(
    TestUmaReadback, "test_numpy_view_device_local_raises", test_numpy_view_device_local_raises, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
