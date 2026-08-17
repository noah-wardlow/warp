#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build Warp for AMD HIP/ROCm.
#
# Environment variables:
#   HIP_ARCH   - comma-separated target architectures (default: gfx1151)
#   ROCM_PATH  - ROCm install path (default: /opt/rocm)
#
# Additional arguments are forwarded to build_lib.py.

set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIP_ARCH="${HIP_ARCH:-gfx1151}"

if [ ! -d "$ROCM_PATH" ]; then
    echo "error: ROCm not found at $ROCM_PATH" >&2
    exit 1
fi

# Resolve a Python 3 interpreter. Many systems only ship `python3` (no bare
# `python`), so don't assume `python` exists. Override with PYTHON=... (e.g.
# to point at a conda/venv interpreter that has the build deps installed:
# numpy, setuptools, packaging, wheel).
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if command -v python >/dev/null 2>&1 && python -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' 2>/dev/null; then
        PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        echo "error: no Python 3 interpreter found (set PYTHON=/path/to/python)" >&2
        exit 1
    fi
fi

"$PYTHON" build_lib.py \
    --no-cuda \
    --rocm-path="$ROCM_PATH" \
    --hip-arch="$HIP_ARCH" \
    --quick \
    "$@"
