<!--
STATUS: FILED at https://github.com/ROCm/ROCm/issues/6398 (ROCm/ROCm umbrella
tracker), 2026-07-05. The issue text below is intentionally framework-neutral.
Until conditional graph nodes ship, the interim downstream
solution is a capability-gated fixed-iteration fallback; that capability check
auto-upgrades to the native path once this lands.
-->

# [Feature request] Conditional graph nodes in HIP graphs (parity with CUDA `cudaGraphNodeTypeConditional`)

## Summary

Please add conditional (IF / WHILE) graph nodes to HIP graphs — parity with CUDA's
conditional graph nodes (12.3 IF / 12.4 WHILE) — so that data-dependent control
flow can be captured *inside* a HIP graph and executed on-device, without a host
round-trip.

## The gap

HIP's `hipGraphNodeType` enum has no conditional node type, and there is no
conditional-handle API — on ROCm 7.2 and on the ROCm/clr `develop` branch:

```console
$ grep -rniE "hipGraphConditional|hipGraphNodeTypeConditional" /opt/rocm/include/
# (no matches)
```

AMD's own HIPIFY CUDA→HIP table confirms it: `cudaGraphConditionalHandleCreate`
(CUDA 12.3) has an **empty HIP column**. The generic `cudaGraphAddNode` maps to
`hipGraphAddNode` (6.2.0), but neither the conditional node type nor the handle it
needs exists.

CUDA provides:

- `cudaGraphConditionalHandleCreate` / `cudaGraphConditionalHandle_t`
- `cudaGraphNodeTypeConditional` with `cudaConditionalNodeParams`
  (`cudaGraphCondTypeIf`, `cudaGraphCondTypeWhile`)
- device-side `cudaGraphSetConditional(handle, value)`
- driver-API equivalents (`cuGraphConditionalHandleCreate`,
  `CU_GRAPH_NODE_TYPE_CONDITIONAL`)

## Requested

- HIP API + runtime support for conditional (IF / WHILE) graph nodes with
  device-set condition values, at CUDA parity.
- A capability query so callers can detect support and choose a portable path.

## Why it matters

A data-dependent loop — e.g. an iterative solver that runs until a convergence
residual is met — cannot be captured inside a HIP graph today. The alternatives
are a fixed iteration count (which over-computes on iterations that would have
converged early) or an eager, non-captured loop (a host round-trip per
iteration). A conditional WHILE node exits at convergence; the fixed fallback
cannot, and its only tuning knob is the iteration cap, which trades accuracy for
speed.

The overhead scales with the iteration budget. For a representative
fixed-iteration GPU loop on gfx1151, throughput is roughly inversely proportional
to the cap (~3.5× between a high and a low iteration count) — headroom that an
early-exiting conditional loop recovers for early-converging steps at no accuracy
cost. In the worst case (a step needs its full budget) the conditional path is
never slower than the fixed one.

## Environment

- ROCm 7.2.1, HIP 7.2
- AMD Radeon 8060S Graphics (gfx1151, RDNA 3.5)

## References

- CUDA conditional graph nodes:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-graph-nodes
- HIPIFY CUDA→HIP table (`cudaGraphConditionalHandleCreate` → no HIP equivalent):
  https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Runtime_API_functions_supported_by_HIP.html
