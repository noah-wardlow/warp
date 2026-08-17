// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "error.h"
#include "texture.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// ============================================================================
// CPU Host Texture Implementation
// ============================================================================

using wp::get_texture_bytes_per_channel;
using wp::Texture;

uint64_t wp_texture_create_host(
    int ndim,
    int num_mip_levels,
    int* mip_widths,
    int* mip_heights,
    int* mip_depths,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    void** mip_data_ptrs_out
)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Warp error: Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS,
            num_mip_levels
        );
        return 0;
    }

    for (int i = 0; i < num_mip_levels; i++) {
        if (mip_widths[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture width must be a positive integer at mip level %d, got %d", i, mip_widths[i]
            );
            return 0;
        }
        if (ndim > 1 && mip_heights[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture height must be a positive integer at mip level %d, got %d", i, mip_heights[i]
            );
            return 0;
        }
        if (ndim > 2 && mip_depths[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture depth must be a positive integer at mip level %d, got %d", i, mip_depths[i]
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    int32_t addr_u = address_modes[0];
    int32_t addr_v = ndim > 1 ? address_modes[1] : 0;
    int32_t addr_w = ndim > 2 ? address_modes[2] : 0;

    Texture* tex = new (wp_alloc_host(sizeof(Texture), "(native:texture)")) Texture(
        ndim, num_mip_levels, mip_widths, mip_heights, mip_depths, num_channels, dtype, filter_mode, mip_filter_mode,
        addr_u, addr_v, addr_w, use_normalized_coords
    );

    if (mip_data_ptrs_out) {
        for (int i = 0; i < num_mip_levels; i++) {
            mip_data_ptrs_out[i] = tex->mip_data[i];
        }
    }

    return reinterpret_cast<uint64_t>(tex);
}

void wp_texture_destroy_host(uint64_t tex_handle)
{
    Texture* tex = (Texture*)tex_handle;
    tex->~Texture();
    wp_free_host(tex);
}

#if WP_ENABLE_CUDA

// ============================================================================
// CUDA Device Texture Implementation
// ============================================================================

#include "cuda_util.h"

// Helper function to get CUDA array format from dtype
static CUarray_format get_cuda_format(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return CU_AD_FORMAT_UNSIGNED_INT8;
    case WP_TEXTURE_DTYPE_UINT16:
        return CU_AD_FORMAT_UNSIGNED_INT16;
    case WP_TEXTURE_DTYPE_UINT32:
        return CU_AD_FORMAT_UNSIGNED_INT32;
    case WP_TEXTURE_DTYPE_INT8:
        return CU_AD_FORMAT_SIGNED_INT8;
    case WP_TEXTURE_DTYPE_INT16:
        return CU_AD_FORMAT_SIGNED_INT16;
    case WP_TEXTURE_DTYPE_INT32:
        return CU_AD_FORMAT_SIGNED_INT32;
    case WP_TEXTURE_DTYPE_FLOAT16:
        return CU_AD_FORMAT_HALF;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return CU_AD_FORMAT_FLOAT;
    }
}

// Helper function to convert address mode int to CUDA address mode
static CUaddress_mode get_cuda_address_mode(int address_mode)
{
    switch (address_mode) {
    case 0:
        return CU_TR_ADDRESS_MODE_WRAP;
    case 1:
        return CU_TR_ADDRESS_MODE_CLAMP;
    case 2:
        return CU_TR_ADDRESS_MODE_MIRROR;
    case 3:
        return CU_TR_ADDRESS_MODE_BORDER;
    default:
        return CU_TR_ADDRESS_MODE_CLAMP;
    }
}

uint64_t wp_texture_create_device(
    void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access, int num_mip_levels
)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture dimensions must be positive integers, got %d at dimension %d", shape[i], i
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Warp error: Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS,
            num_mip_levels
        );
        return 0;
    }

    ContextGuard guard(context);

    size_t width = shape[0];
    size_t height = ndim > 1 ? shape[1] : 0;
    size_t depth = ndim > 2 ? shape[2] : 0;

    CUarray_format format = get_cuda_format(dtype);

    CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Depth = depth;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;
    arr_desc.Flags = surface_access ? CUDA_ARRAY3D_SURFACE_LDST : 0;

    if (num_mip_levels > 1) {
        CUmipmappedArray mipmap_array = NULL;
        if (!check_cu(cuMipmappedArrayCreate_f(&mipmap_array, &arr_desc, (unsigned int)num_mip_levels))) {
            return 0;
        }
        return reinterpret_cast<uint64_t>(mipmap_array);
    }

    CUarray cuda_array = NULL;
    if (!check_cu(cuArray3DCreate_f(&cuda_array, &arr_desc))) {
        return 0;
    }
    return reinterpret_cast<uint64_t>(cuda_array);
}

void wp_texture_destroy_device(void* context, uint64_t array_handle, bool is_mipmapped)
{
    ContextGuard guard(context);
    if (array_handle == 0) {
        return;
    }
    if (is_mipmapped) {
        check_cu(cuMipmappedArrayDestroy_f((CUmipmappedArray)array_handle));
    } else {
        check_cu(cuArrayDestroy_f((CUarray)array_handle));
    }
}

uint64_t wp_texture_get_mip_level_array_device(void* context, uint64_t mipmap_array_handle, int level)
{
    if (!mipmap_array_handle) {
        wp::set_error_string("Warp error: Null mipmapped array handle");
        return 0;
    }

    ContextGuard guard(context);

    CUarray level_array = NULL;
    if (!check_cu(
            cuMipmappedArrayGetLevel_f(&level_array, (CUmipmappedArray)mipmap_array_handle, (unsigned int)level)
        )) {
        return 0;
    }
    return reinterpret_cast<uint64_t>(level_array);
}

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned width_texels,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    (void)width_texels;  // only used by the HIP 3D runtime-copy path below
    ContextGuard guard(context);

    CUstream cuda_stream = static_cast<CUstream>(stream);

    CUDA_MEMCPY3D copy_params = {};
    copy_params.WidthInBytes = width_bytes;
    copy_params.Height = height;
    copy_params.Depth = depth;

    // dst/src_memory_type arrive in the CUmemorytype *driver* convention used by
    // the Python MemoryType enum (HOST=1, DEVICE=2, ARRAY=3). On HIP the
    // hipMemoryType enum uses different numeric values, so compare against the
    // driver convention explicitly and assign the backend enum (CU_MEMORYTYPE_*,
    // which resolves to hipMemoryType* on HIP) rather than casting the raw int.
    enum { WP_MEMTYPE_HOST = 1, WP_MEMTYPE_DEVICE = 2, WP_MEMTYPE_ARRAY = 3 };

#if defined(__HIP_PLATFORM_AMD__)
    // On gfx1151 / ROCm 7.2 the HIP host<->array texture copy corrupts the leading
    // texels of the array: the first ~16 bytes (one texel block) read back stale,
    // in a size- and allocation-dependent (partly non-deterministic) pattern.
    // Verified experimentally that the *device*<->array path is clean while the
    // *host*<->array path is not, so stage any host endpoint through a temporary
    // device buffer:
    //   HOST->ARRAY : host -> device(linear) -> array
    //   ARRAY->HOST : array -> device(linear) -> host
    // Device<->array copies go direct. 1D/2D use the runtime hipMemcpy2D{To,From}Array
    // entry points; 3D uses the driver hipDrvMemcpy3D (also clean with a device endpoint).
    // Only copies with exactly one array endpoint are handled; array<->array falls through.
    if (depth <= 1 && ((dst_memory_type == WP_MEMTYPE_ARRAY) != (src_memory_type == WP_MEMTYPE_ARRAY))) {
        const size_t rows = height > 0 ? height : 1;
        const size_t total = size_t(width_bytes) * rows;

        if (dst_memory_type == WP_MEMTYPE_ARRAY) {
            if (src_memory_type == WP_MEMTYPE_DEVICE) {
                return check_cuda(hipMemcpy2DToArrayAsync(
                    reinterpret_cast<hipArray_t>(dst_handle), 0, 0, reinterpret_cast<const void*>(src_handle),
                    src_pitch, width_bytes, rows, hipMemcpyDeviceToDevice, cuda_stream
                ));
            }
            // HOST -> DEVICE(staging) -> ARRAY
            void* staging = nullptr;
            if (hipMalloc(&staging, total) != hipSuccess) {
                wp::set_error_string("Warp error: failed to allocate texture staging buffer (%zu bytes)", total);
                return false;
            }
            hipError_t e = hipMemcpy2DAsync(
                staging, width_bytes, reinterpret_cast<const void*>(src_handle), src_pitch, width_bytes, rows,
                hipMemcpyHostToDevice, cuda_stream
            );
            if (e == hipSuccess) {
                e = hipMemcpy2DToArrayAsync(
                    reinterpret_cast<hipArray_t>(dst_handle), 0, 0, staging, width_bytes, width_bytes, rows,
                    hipMemcpyDeviceToDevice, cuda_stream
                );
            }
            hipError_t sync_err = hipStreamSynchronize(cuda_stream);
            hipError_t free_err = hipFree(staging);
            if (e == hipSuccess)
                e = (sync_err != hipSuccess) ? sync_err : free_err;
            return check_cuda(e);
        } else {
            if (dst_memory_type == WP_MEMTYPE_DEVICE) {
                return check_cuda(hipMemcpy2DFromArrayAsync(
                    reinterpret_cast<void*>(dst_handle), dst_pitch, reinterpret_cast<hipArray_const_t>(src_handle), 0,
                    0, width_bytes, rows, hipMemcpyDeviceToDevice, cuda_stream
                ));
            }
            // ARRAY -> DEVICE(staging) -> HOST
            void* staging = nullptr;
            if (hipMalloc(&staging, total) != hipSuccess) {
                wp::set_error_string("Warp error: failed to allocate texture staging buffer (%zu bytes)", total);
                return false;
            }
            hipError_t e = hipMemcpy2DFromArrayAsync(
                staging, width_bytes, reinterpret_cast<hipArray_const_t>(src_handle), 0, 0, width_bytes, rows,
                hipMemcpyDeviceToDevice, cuda_stream
            );
            if (e == hipSuccess) {
                e = hipMemcpy2DAsync(
                    reinterpret_cast<void*>(dst_handle), dst_pitch, staging, width_bytes, width_bytes, rows,
                    hipMemcpyDeviceToHost, cuda_stream
                );
            }
            hipError_t sync_err = hipStreamSynchronize(cuda_stream);
            hipError_t free_err = hipFree(staging);
            if (e == hipSuccess)
                e = (sync_err != hipSuccess) ? sync_err : free_err;
            return check_cuda(e);
        }
    }

    // 3D textures: same host<->array corruption, AND the driver hipDrvMemcpy3D is not
    // recorded into HIP graphs (device<->array copies under capture replay as no-ops).
    // Stage any host endpoint through a temporary device buffer, then run the *runtime*
    // hipMemcpy3DAsync for the device<->array leg -- it is capturable and honors the
    // array pitch. The runtime extent is in array elements, so read the element width
    // from the array descriptor. Device-direct copies stay fully async (no alloc/sync/
    // free) so they remain legal and get captured during graph capture.
    if (depth > 1 && ((dst_memory_type == WP_MEMTYPE_ARRAY) != (src_memory_type == WP_MEMTYPE_ARRAY))) {
        const size_t rows = height > 0 ? height : 1;
        const size_t total = size_t(width_bytes) * rows * size_t(depth);
        const bool to_array = (dst_memory_type == WP_MEMTYPE_ARRAY);
        const uint64_t array_handle = to_array ? dst_handle : src_handle;
        const int linear_mt = to_array ? src_memory_type : dst_memory_type;
        const uint64_t linear_handle = to_array ? src_handle : dst_handle;
        const unsigned linear_pitch = to_array ? src_pitch : dst_pitch;

        void* staging = nullptr;
        void* dev_ptr;
        if (linear_mt == WP_MEMTYPE_DEVICE) {
            dev_ptr = reinterpret_cast<void*>(linear_handle);
        } else {
            if (hipMalloc(&staging, total) != hipSuccess) {
                wp::set_error_string("Warp error: failed to allocate texture staging buffer (%zu bytes)", total);
                return false;
            }
            dev_ptr = staging;
        }

        hipError_t e = hipSuccess;
        // host -> device staging (upload) before the array copy
        if (staging && to_array) {
            e = hipMemcpy2DAsync(
                staging, width_bytes, reinterpret_cast<const void*>(linear_handle), linear_pitch, width_bytes,
                rows * depth, hipMemcpyHostToDevice, cuda_stream
            );
        }

        if (e == hipSuccess) {
            // Runtime hipMemcpy3D measures the array extent in *elements* (texels),
            // so use the texel width passed from Python. Querying the array
            // descriptor here would be illegal during graph capture.
            hipMemcpy3DParms p = {};
            p.extent = make_hipExtent(width_texels, rows, depth);
            p.kind = hipMemcpyDeviceToDevice;
            if (to_array) {
                p.srcPtr = make_hipPitchedPtr(dev_ptr, width_bytes, width_bytes, rows);
                p.dstArray = reinterpret_cast<hipArray_t>(array_handle);
            } else {
                p.srcArray = reinterpret_cast<hipArray_t>(array_handle);
                p.dstPtr = make_hipPitchedPtr(dev_ptr, width_bytes, width_bytes, rows);
            }
            e = hipMemcpy3DAsync(&p, cuda_stream);
        }

        // device staging -> host (readback) after the array copy
        if (staging && !to_array && e == hipSuccess) {
            e = hipMemcpy2DAsync(
                reinterpret_cast<void*>(linear_handle), linear_pitch, staging, width_bytes, width_bytes,
                rows * depth, hipMemcpyDeviceToHost, cuda_stream
            );
        }

        if (staging) {
            hipError_t sync_err = hipStreamSynchronize(cuda_stream);
            hipError_t free_err = hipFree(staging);
            if (e == hipSuccess)
                e = (sync_err != hipSuccess) ? sync_err : free_err;
        }
        return check_cuda(e);
    }
#endif  // defined(__HIP_PLATFORM_AMD__)

    if (dst_memory_type == WP_MEMTYPE_HOST) {
        copy_params.dstMemoryType = CU_MEMORYTYPE_HOST;
        copy_params.dstHost = reinterpret_cast<void*>(dst_handle);
    } else if (dst_memory_type == WP_MEMTYPE_DEVICE) {
        copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy_params.dstDevice = (CUdeviceptr)(dst_handle);
    } else if (dst_memory_type == WP_MEMTYPE_ARRAY) {
        copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy_params.dstArray = reinterpret_cast<CUarray>(dst_handle);
    } else {
        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
        return false;
    }
    copy_params.dstPitch = dst_pitch;
    copy_params.dstHeight = dst_height;

    if (src_memory_type == WP_MEMTYPE_HOST) {
        copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy_params.srcHost = reinterpret_cast<void*>(src_handle);
    } else if (src_memory_type == WP_MEMTYPE_DEVICE) {
        copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy_params.srcDevice = (CUdeviceptr)(src_handle);
    } else if (src_memory_type == WP_MEMTYPE_ARRAY) {
        copy_params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copy_params.srcArray = reinterpret_cast<CUarray>(src_handle);
    } else {
        wp::set_error_string("Invalid source memory type %d", src_memory_type);
        return false;
    }
    copy_params.srcPitch = src_pitch;
    copy_params.srcHeight = src_height;

    return check_cu(cuMemcpy3DAsync_f(&copy_params, cuda_stream));
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    if (!array_handle || !desc_out) {
        wp::set_error_string("Warp error: NULL array handle");
        return false;
    }

    ContextGuard guard(context);

    CUarray cuda_array = reinterpret_cast<CUarray>(array_handle);

    CUDA_ARRAY3D_DESCRIPTOR desc;
    if (!check_cu(cuArray3DGetDescriptor_f(&desc, cuda_array)))
        return false;

    switch (desc.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT8;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT16;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT32;
        break;
    case CU_AD_FORMAT_SIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT8;
        break;
    case CU_AD_FORMAT_SIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT16;
        break;
    case CU_AD_FORMAT_SIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT32;
        break;
    case CU_AD_FORMAT_HALF:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT16;
        break;
    case CU_AD_FORMAT_FLOAT:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT32;
        break;
    default:
        wp::set_error_string("Warp error: Unsupported texture format");
        return false;
    }

    desc_out->shape[0] = int32_t(desc.Width);
    desc_out->shape[1] = int32_t(desc.Height);
    desc_out->shape[2] = int32_t(desc.Depth);

    if (desc.Depth == 0) {
        if (desc.Height == 0)
            desc_out->ndim = 1;
        else
            desc_out->ndim = 2;
    } else {
        desc_out->ndim = 3;
    }

    // TODO: desc.Flags?

    desc_out->num_channels = int32_t(desc.NumChannels);

    return true;
}

uint64_t wp_texture_object_create_device(
    void* context,
    uint64_t array_handle,
    int ndim,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    int num_mip_levels
)
{
    if (!array_handle) {
        wp::set_error_string("Null texture array handle");
        return 0;
    }

    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    if (num_mip_levels < 1 || num_mip_levels > WP_TEXTURE_MAX_MIP_LEVELS) {
        wp::set_error_string(
            "Number of texture mip levels must be in [1, %d], got %d", WP_TEXTURE_MAX_MIP_LEVELS, num_mip_levels
        );
        return 0;
    }

    ContextGuard guard(context);

    CUDA_RESOURCE_DESC res_desc = {};
    if (num_mip_levels > 1) {
        res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        res_desc.res.mipmap.hMipmappedArray = reinterpret_cast<CUmipmappedArray>(array_handle);
    } else {
        res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = reinterpret_cast<CUarray>(array_handle);
    }
    res_desc.flags = 0;

    CUDA_TEXTURE_DESC tex_desc = {};

    for (int i = 0; i < 3; i++) {
        if (i < ndim)
            tex_desc.addressMode[i] = get_cuda_address_mode(address_modes[i]);
        else
            tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_CLAMP;
    }

    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;

    // Coordinate mode: normalized [0,1] or texel space [0,width/height]
    // For uint8/uint16 textures, CUDA automatically normalizes values to [0,1] when sampled
    // (since CU_TRSF_READ_AS_INTEGER is NOT set). Float32 textures return values as-is.
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;

    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapFilterMode = (mip_filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = (num_mip_levels > 1) ? (float)(num_mip_levels - 1) : 0.0f;

    CUtexObject tex_object = 0;
    check_cu(cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr));

    return (uint64_t)(tex_object);
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle)
{
    ContextGuard guard(context);

    if (tex_handle != 0) {
        check_cu(cuTexObjectDestroy_f((CUtexObject)tex_handle));
    }
}

uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    ContextGuard guard(context);

    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = reinterpret_cast<cudaArray_t>(array_handle);

    cudaSurfaceObject_t surface = 0;
    check_cuda(cudaCreateSurfaceObject(&surface, &desc));

    return (uint64_t)(surface);
}

void wp_surface_object_destroy_device(void* context, uint64_t surface_handle)
{
    if (!surface_handle)
        return;

    ContextGuard guard(context);
    check_cuda(cudaDestroySurfaceObject((cudaSurfaceObject_t)surface_handle));
}

#else  // WP_ENABLE_CUDA

// Stub implementations for non-CUDA builds

uint64_t wp_texture_create_device(
    void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access, int num_mip_levels
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_destroy_device(void* context, uint64_t array_handle, bool is_mipmapped) { }

uint64_t wp_texture_get_mip_level_array_device(void* context, uint64_t mipmap_array_handle, int level)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned width_texels,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

uint64_t wp_texture_object_create_device(
    void* context,
    uint64_t array_handle,
    int ndim,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    int num_mip_levels
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle) { }

WP_API uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

WP_API void wp_surface_object_destroy_device(void* context, uint64_t surface_handle) { }

#endif  // WP_ENABLE_CUDA
