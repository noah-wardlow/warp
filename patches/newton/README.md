# Newton HIP/ROCm patches for AMD gfx1151 (Strix Halo)

Local Newton source patches for running Newton examples on this fork.

# Patch 03: contact_reduction HIP device guards

`newton/_src/geometry/contact_reduction.py` embeds native C++ snippets (via
`@wp.func_native`) gated on `#if defined(__CUDA_ARCH__)` only. HIP device
compilation defines `__HIP_DEVICE_COMPILE__` instead, so on AMD GPUs the `#else`
branches compile into every contact-reduction kernel:

- `synchronize()` compiles to a no-op (no `__syncthreads()` barrier), and
- `get_shared_memory_pointer()` (three variants) returns `NULL`,

so any narrow-phase kernel using contact reduction writes to address zero
(an illegal memory access on the first colliding frame).

## Affected examples (all mesh narrow-phase with `reduce_contacts=True`)

basic_conveyor, basic_shapes, cable_ball_joints, cable_fixed_joints,
cloth_franka, cloth_h1, cloth_style3d, ik_cube_stacking, nut_bolt_hydro,
nut_bolt_sdf, robot_panda_hydro

## Fix

Add `|| defined(__HIP_DEVICE_COMPILE__)` to all four `#if` guards. The CUDA path
is unchanged. No warp rebuild needed — the snippet change alters the kernel
module hashes and JIT recompiles on next run.

## How to apply

```bash
NEWTON_DIR=$(python -c "import newton, os; print(os.path.dirname(newton.__file__))")
cd "$NEWTON_DIR"
patch -p2 < /path/to/warp/patches/newton/03-contact_reduction-hip-device-guards.patch
```

---

# Patch 04: MuJoCo solver — disable conditional graph nodes on HIP

Newton's `SolverMuJoCo` leaves mujoco-warp's `graph_conditional` option at its
default `True`, so the solver uses `wp.capture_while` (CUDA 12.4+ conditional
graph nodes) for its iterative convergence loop. HIP/ROCm has no conditional
graph node API (no `hipGraphConditionalHandle` through ROCm 7.2), so capturing a
simulation graph that includes `SolverMuJoCo.step` raises
`Conditional graph nodes are not supported on HIP/ROCm`, and the stock MuJoCo
examples (`robot_anymal_c_walk`, cartpole, humanoid, …) cannot run.

## Fix

In `SolverMuJoCo.__init__`, set `mjw_model.opt.graph_conditional = False` when
any CUDA device is HIP. mujoco-warp then uses a fixed-iteration solve (no
conditional nodes), which is both capturable and faster in eager stepping (no
per-iteration device→host sync). This makes the stock examples run directly with
no wrapper. The CUDA path is unchanged (the guard is a no-op when no HIP device
is present).

The fix gates on `wp.is_conditional_graph_supported()`, so it uses the
conditional path automatically on any backend that supports conditional graph
nodes. The root-cause fix is at the platform level (HIP has no conditional graph
node API); a drafted feature request to add it — parity with CUDA's
`cudaGraphNodeTypeConditional` — is in `rocm-conditional-graph-feature-request.md`.

## How to apply

```bash
NEWTON_DIR=$(python -c "import newton, os; print(os.path.dirname(newton.__file__))")
cd "$NEWTON_DIR/_src"
patch -p3 < /path/to/warp/patches/newton/04-solver_mujoco-hip-graph-conditional.patch
```

## Verified results (Newton on gfx1151, ROCm 7.2)

`robot_anymal_c_walk` (headless, 400 frames): stable ~31 FPS with 0.0 MB
GPU-memory growth.
