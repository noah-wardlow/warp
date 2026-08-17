# SPDX-License-Identifier: Apache-2.0
"""Zero-copy readback helpers for unified-memory (APU / iGPU) devices.

On a unified-memory device (e.g. an AMD APU / gfx1151 iGPU) the CPU and GPU
share physical DRAM, yet a plain device-local buffer is coarse-grained and
uncached from the CPU, so :meth:`warp.array.numpy` must pay a slow device->host
copy to read it back. These helpers instead place the buffer in CPU-accessible
unified (managed) memory, so a result can be read back as a **zero-copy NumPy
view** of the same physical memory -- no copy. Measured ~2-4x faster than
``numpy()`` for large readbacks on gfx1151.

On non-unified-memory devices (a discrete GPU) or the CPU they transparently
fall back to a normal copying readback, so code stays portable.

Trade-off: managed memory is slower than device-local memory for bandwidth-bound
GPU kernels. Keep compute buffers device-local (the default) and use these
helpers only for buffers you read back to the CPU (observations, rewards,
logging, visualization).
"""

import ctypes

import numpy as np

import warp._src.context as _context
import warp._src.types as _types


def _view_spec(wdtype, shape):
    """(numpy dtype, full shape incl. component dims, element count) for a
    normalized Warp dtype and logical shape."""
    if hasattr(wdtype, "_wp_scalar_type_"):  # vector / matrix / quaternion
        scalar = wdtype._wp_scalar_type_
        comp = tuple(int(d) for d in wdtype._shape_)
    else:
        scalar = wdtype
        comp = ()
    np_dtype = np.dtype(_types.warp_type_to_np_dtype[scalar])
    full_shape = tuple(int(s) for s in shape) + comp
    count = 1
    for s in full_shape:
        count *= int(s)
    return np_dtype, full_shape, count


def zero_copy_view(ptr, wdtype, shape):
    """Build a zero-copy NumPy array aliasing CPU-accessible memory at ``ptr``.

    The caller is responsible for ensuring the memory is CPU-accessible
    (unified / managed) and that outstanding GPU writes have been synchronized.
    The returned array aliases the memory and is valid only while it stays live.
    """
    np_dtype, full_shape, count = _view_spec(wdtype, shape)
    nbytes = count * np_dtype.itemsize
    raw = (ctypes.c_byte * nbytes).from_address(int(ptr))
    return np.frombuffer(raw, dtype=np_dtype, count=count).reshape(full_shape)


class ReadbackBuffer:
    """CPU-accessible staging buffer for fast zero-copy GPU->CPU readback on
    unified-memory (APU / iGPU) devices.

    Write your GPU result into :attr:`array` (device-side: :func:`warp.copy` or a
    kernel launch), then call :meth:`numpy` to read it. On unified memory this
    returns a **zero-copy view** of the same physical memory (no device->host
    copy) -- ~2-4x faster than ``array.numpy()`` for large readbacks on gfx1151.
    On non-unified devices it falls back to a device buffer plus a copying
    readback, so the same code runs everywhere.

    Example::

        buf = wp.ReadbackBuffer(shape=(n,), dtype=wp.float32, device="cuda:0")
        # per step: write the obs the CPU needs into the managed buffer
        wp.copy(buf.array, device_local_result)  # or wp.launch(..., outputs=[buf.array])
        obs = buf.numpy()  # zero-copy view on an APU

    The array returned by :meth:`numpy` **aliases** device memory on unified
    hardware: it is valid only while this buffer is alive and not being written
    by the GPU (``numpy()`` synchronizes the device first).

    Args:
        shape: Buffer shape (int or tuple).
        dtype: Warp data type (scalar, vector, or matrix). Defaults to ``float``.
        device: Device to allocate on. Defaults to the current device.
    """

    def __init__(self, shape, dtype=float, device=None):
        import warp

        self.device = warp.get_device(device)
        self.shape = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        self.dtype = dtype
        wdtype = _types.type_to_warp(dtype)
        count = 1
        for s in self.shape:
            count *= int(s)
        self._nbytes = count * _types.type_size_in_bytes(wdtype)
        self._ptr = 0

        # Only unified-memory CUDA/HIP devices can be read back zero-copy.
        self.zero_copy = bool(self.device.is_cuda and getattr(self.device, "is_uma", False))
        if self.zero_copy:
            self._ptr = int(_context.runtime.core.wp_alloc_managed_device(self.device.context, self._nbytes))
            if self._ptr:
                # Coarse-grained, so a kernel writing this buffer with the hardware
                # float atomic is not silently discarded. Readback synchronizes
                # first, which is where coarse-grained memory becomes host-visible.
                if not _context.runtime.core.wp_cuda_set_memory_coarse_grain(
                    self.device.context, self._ptr, self._nbytes
                ):
                    from warp._src.logger import log_warning  # noqa: PLC0415

                    log_warning(
                        f"Failed to mark a {self._nbytes}-byte ReadbackBuffer coarse-grained on device "
                        f"'{self.device}'. Float32 atomics written into it are silently discarded while "
                        f"warp.config.hip_fast_fp_atomics is enabled.",
                        once=True,
                    )
                self.array = warp.array(
                    ptr=self._ptr, dtype=dtype, shape=self.shape, capacity=self._nbytes, device=self.device
                )
                self._view = zero_copy_view(self._ptr, wdtype, self.shape)
            else:
                self.zero_copy = False  # managed alloc failed -> fall back

        if not self.zero_copy:
            self.array = warp.zeros(self.shape, dtype=dtype, device=self.device)
            self._view = None

    def numpy(self):
        """Return the buffer contents as a NumPy array, synchronizing first.

        Zero-copy view on unified memory (no device->host copy); an ordinary
        copying readback otherwise.
        """
        import warp

        warp.synchronize_device(self.device)
        return self._view if self.zero_copy else self.array.numpy()

    def __len__(self):
        return self.shape[0]

    def release(self):
        """Free the underlying managed allocation (idempotent)."""
        if self._ptr:
            _context.runtime.core.wp_free_managed_device(self.device.context, self._ptr)
            self._ptr = 0
            self._view = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
