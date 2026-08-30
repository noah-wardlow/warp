Enable the bundled cuBQL GPU BVH builder on HIP/ROCm. HIP devices now report
cuBQL availability and support `constructor="cubql"` for BVHs and meshes,
using the same native-layout conversion, query, refit, and rebuild paths as
CUDA.
