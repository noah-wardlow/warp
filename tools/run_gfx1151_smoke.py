# SPDX-License-Identifier: Apache-2.0

import warp as wp


@wp.kernel
def vec_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = a[i] + b[i]


@wp.kernel
def vec_scale(a: wp.array(dtype=float), s: float):
    i = wp.tid()
    a[i] = a[i] * s


def main():
    print("Initializing Warp...")
    wp.init()

    devices = wp.get_devices()
    print("Detected devices:")
    for d in devices:
        print("  ", d, "| arch:", getattr(d, "arch_str", "") or "unknown")

    device = wp.get_preferred_device()
    print("\nUsing device:", device)

    # On HIP the architecture is in arch_str (e.g. "gfx1151"); the integer
    # arch field is 0 there.
    arch = getattr(device, "arch_str", "") or ""
    if arch:
        print("Architecture:", arch)
        if "gfx1151" not in arch.lower():
            print("WARNING: Not running on gfx1151!")
    else:
        print("Architecture: unavailable on this backend")

    n = 1024

    print("\nAllocating arrays...")
    a = wp.array([1.0] * n, dtype=float, device=device)
    b = wp.array([2.0] * n, dtype=float, device=device)
    c = wp.zeros(n, dtype=float, device=device)

    print("Running vec_add...")
    wp.launch(vec_add, dim=n, inputs=[a, b, c], device=device)

    result = c.numpy()
    print("vec_add result sample:", result[:5])

    if not all(abs(x - 3.0) < 1e-6 for x in result):
        raise RuntimeError("vec_add failed")

    print("Running vec_scale...")
    wp.launch(vec_scale, dim=n, inputs=[c, 2.0], device=device)

    result = c.numpy()
    print("vec_scale result sample:", result[:5])

    if not all(abs(x - 6.0) < 1e-6 for x in result):
        raise RuntimeError("vec_scale failed")

    print("\n✅ GFX1151 smoke test PASSED")


if __name__ == "__main__":
    main()
