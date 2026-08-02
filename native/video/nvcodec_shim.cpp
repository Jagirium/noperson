#include "video_ffi.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <deque>
#include <new>
#include <vector>

#if defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
#define FFNV_LOG_FUNC(logctx, msg, ...) ((void)(logctx))
#define FFNV_DEBUG_LOG_FUNC(logctx, msg, ...) ((void)(logctx))
#include <ffnvcodec/dynlink_loader.h>
#endif

struct NpNvDecoder {
#if defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    CudaFunctions *cuda = nullptr;
    CuvidFunctions *cuvid = nullptr;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUvideoparser parser = nullptr;
    CUvideodecoder decoder = nullptr;
    std::deque<CUVIDPARSERDISPINFO> ready;
    std::vector<CUdeviceptr> mapped;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pixel_format = 0;
    CUresult callback_error = CUDA_SUCCESS;
#endif
};

namespace {

int32_t invalid_probe(NpVideoError *error) noexcept {
    if (error != nullptr) {
        error->code = -EINVAL;
        std::snprintf(error->message, sizeof(error->message),
                      "invalid NVCodec probe output pointer");
    }
    return -EINVAL;
}

#if defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
cudaVideoCodec video_codec(int32_t codec) noexcept {
    switch (codec) {
    case 1:
        return cudaVideoCodec_H264;
    case 2:
        return cudaVideoCodec_HEVC;
    case 3:
        return cudaVideoCodec_AV1;
    case 4:
        return cudaVideoCodec_VP9;
    default:
        return cudaVideoCodec_NumCodecs;
    }
}

int32_t cuda_fail(NpNvDecoder *decoder, NpVideoError *error, CUresult code,
                  const char *context) noexcept {
    if (error != nullptr) {
        error->code = static_cast<int32_t>(code);
        const char *detail = nullptr;
        if (decoder != nullptr && decoder->cuda != nullptr &&
            decoder->cuda->cuGetErrorString != nullptr) {
            decoder->cuda->cuGetErrorString(code, &detail);
        }
        std::snprintf(error->message, sizeof(error->message), "%s: %s", context,
                      detail != nullptr ? detail : "CUDA/NVDEC failure");
    }
    return -static_cast<int32_t>(code == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : code);
}

bool push_context(NpNvDecoder *decoder, NpVideoError *error) noexcept {
    const CUresult status = decoder->cuda->cuCtxPushCurrent(decoder->context);
    if (status != CUDA_SUCCESS) {
        cuda_fail(decoder, error, status, "activate CUDA context");
        return false;
    }
    return true;
}

int32_t context_failure(const NpVideoError *error) noexcept {
    if (error != nullptr && error->code != 0) {
        return error->code < 0 ? error->code : -error->code;
    }
    return -static_cast<int32_t>(CUDA_ERROR_UNKNOWN);
}

void pop_context(NpNvDecoder *decoder) noexcept {
    CUcontext popped = nullptr;
    decoder->cuda->cuCtxPopCurrent(&popped);
}

int CUDAAPI decoder_sequence(void *opaque, CUVIDEOFORMAT *format) {
    auto *decoder = static_cast<NpNvDecoder *>(opaque);
    if (decoder->decoder != nullptr) {
        decoder->callback_error = CUDA_ERROR_UNKNOWN;
        return 0;
    }
    CUVIDDECODECREATEINFO create{};
    create.ulWidth = format->coded_width;
    create.ulHeight = format->coded_height;
    create.ulNumDecodeSurfaces = format->min_num_decode_surfaces;
    create.CodecType = format->codec;
    create.ChromaFormat = format->chroma_format;
    create.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    create.bitDepthMinus8 = format->bit_depth_luma_minus8;
    create.ulMaxWidth = format->coded_width;
    create.ulMaxHeight = format->coded_height;
    create.display_area.left = static_cast<short>(format->display_area.left);
    create.display_area.top = static_cast<short>(format->display_area.top);
    create.display_area.right = static_cast<short>(format->display_area.right);
    create.display_area.bottom = static_cast<short>(format->display_area.bottom);
    create.OutputFormat = format->bit_depth_luma_minus8 > 0
                              ? cudaVideoSurfaceFormat_P016
                              : cudaVideoSurfaceFormat_NV12;
    create.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    decoder->width = static_cast<uint32_t>(format->display_area.right - format->display_area.left);
    decoder->height = static_cast<uint32_t>(format->display_area.bottom - format->display_area.top);
    create.ulTargetWidth = decoder->width;
    create.ulTargetHeight = decoder->height;
    create.ulNumOutputSurfaces = 8;
    create.target_rect.right = static_cast<short>(decoder->width);
    create.target_rect.bottom = static_cast<short>(decoder->height);
    decoder->pixel_format = format->bit_depth_luma_minus8 > 0 ? 2U : 1U;
    decoder->callback_error = decoder->cuvid->cuvidCreateDecoder(&decoder->decoder, &create);
    return decoder->callback_error == CUDA_SUCCESS
               ? static_cast<int>(format->min_num_decode_surfaces)
               : 0;
}

int CUDAAPI decoder_decode(void *opaque, CUVIDPICPARAMS *picture) {
    auto *decoder = static_cast<NpNvDecoder *>(opaque);
    decoder->callback_error = decoder->cuvid->cuvidDecodePicture(decoder->decoder, picture);
    return decoder->callback_error == CUDA_SUCCESS ? 1 : 0;
}

int CUDAAPI decoder_display(void *opaque, CUVIDPARSERDISPINFO *display) {
    auto *decoder = static_cast<NpNvDecoder *>(opaque);
    if (display == nullptr) {
        return 1;
    }
    try {
        decoder->ready.push_back(*display);
        return 1;
    } catch (...) {
        decoder->callback_error = CUDA_ERROR_UNKNOWN;
        return 0;
    }
}

void destroy_decoder(NpNvDecoder *decoder) noexcept {
    if (decoder == nullptr) {
        return;
    }
    if (decoder->cuda != nullptr && decoder->context != nullptr) {
        NpVideoError ignored{};
        if (push_context(decoder, &ignored)) {
            for (const CUdeviceptr pointer : decoder->mapped) {
                decoder->cuvid->cuvidUnmapVideoFrame(decoder->decoder, pointer);
            }
            if (decoder->parser != nullptr) {
                decoder->cuvid->cuvidDestroyVideoParser(decoder->parser);
            }
            if (decoder->decoder != nullptr) {
                decoder->cuvid->cuvidDestroyDecoder(decoder->decoder);
            }
            pop_context(decoder);
        }
    }
    cuvid_free_functions(&decoder->cuvid);
    cuda_free_functions(&decoder->cuda);
    delete decoder;
}
#else
void destroy_decoder(NpNvDecoder *decoder) noexcept { delete decoder; }
#endif

} // namespace

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvcodec_probe(
    NpNvCodecCapabilities *out_capabilities,
    NpVideoError *error) noexcept {
    if (out_capabilities == nullptr) {
        return invalid_probe(error);
    }
    if (error != nullptr) {
        error->code = 0;
        error->message[0] = '\0';
    }
    std::memset(out_capabilities, 0, sizeof(*out_capabilities));
    out_capabilities->abi_version = NP_VIDEO_ABI_VERSION;

#if defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    CudaFunctions *cuda = nullptr;
    CuvidFunctions *cuvid = nullptr;
    NvencFunctions *nvenc = nullptr;
    out_capabilities->cuda_driver = cuda_load_functions(&cuda, nullptr) == 0;
    out_capabilities->nvdec = cuvid_load_functions(&cuvid, nullptr) == 0;
    out_capabilities->nvenc = nvenc_load_functions(&nvenc, nullptr) == 0;
    if (out_capabilities->nvenc) {
        uint32_t version = 0;
        if (nvenc->NvEncodeAPIGetMaxSupportedVersion(&version) == NV_ENC_SUCCESS) {
            out_capabilities->nvenc_api_major = version & 0x00ffffffU;
            out_capabilities->nvenc_api_minor = version >> 24U;
        }
    }
    cuda_free_functions(&cuda);
    cuvid_free_functions(&cuvid);
    nvenc_free_functions(&nvenc);
#endif
    return 0;
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvdecoder_open(
    void *cuda_context,
    void *cuda_stream,
    int32_t codec,
    NpNvDecoder **out_decoder,
    NpVideoError *error) noexcept {
    if (error != nullptr) {
        error->code = 0;
        error->message[0] = '\0';
    }
    if (cuda_context == nullptr || out_decoder == nullptr) {
        return invalid_probe(error);
    }
    *out_decoder = nullptr;
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    (void)cuda_stream;
    (void)codec;
    if (error != nullptr) {
        error->code = -ENOSYS;
        std::snprintf(error->message, sizeof(error->message), "NVCodec headers were unavailable at build time");
    }
    return -ENOSYS;
#else
    if (video_codec(codec) == cudaVideoCodec_NumCodecs) {
        if (error != nullptr) {
            error->code = -EINVAL;
            std::snprintf(error->message, sizeof(error->message), "unsupported NVDEC codec");
        }
        return -EINVAL;
    }
    auto *decoder = new (std::nothrow) NpNvDecoder();
    if (decoder == nullptr) {
        return -ENOMEM;
    }
    decoder->context = static_cast<CUcontext>(cuda_context);
    decoder->stream = static_cast<CUstream>(cuda_stream);
    if (cuda_load_functions(&decoder->cuda, nullptr) != 0 ||
        cuvid_load_functions(&decoder->cuvid, nullptr) != 0) {
        if (error != nullptr) {
            error->code = -ENOENT;
            std::snprintf(error->message, sizeof(error->message), "load CUDA/NVDEC driver APIs failed");
        }
        destroy_decoder(decoder);
        return -ENOENT;
    }
    CUVIDPARSERPARAMS parser{};
    parser.CodecType = video_codec(codec);
    parser.ulMaxNumDecodeSurfaces = 8;
    parser.ulMaxDisplayDelay = 2;
    parser.pUserData = decoder;
    parser.pfnSequenceCallback = decoder_sequence;
    parser.pfnDecodePicture = decoder_decode;
    parser.pfnDisplayPicture = decoder_display;
    const CUresult status = decoder->cuvid->cuvidCreateVideoParser(&decoder->parser, &parser);
    if (status != CUDA_SUCCESS) {
        const int32_t result = cuda_fail(decoder, error, status, "create NVDEC parser");
        destroy_decoder(decoder);
        return result;
    }
    *out_decoder = decoder;
    return 0;
#endif
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvdecoder_send(
    NpNvDecoder *decoder,
    const uint8_t *data,
    size_t data_len,
    int64_t timestamp_100ns,
    NpVideoError *error) noexcept {
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    (void)decoder; (void)data; (void)data_len; (void)timestamp_100ns;
    return invalid_probe(error);
#else
    if (decoder == nullptr || (data_len > 0 && data == nullptr)) {
        return invalid_probe(error);
    }
    if (!push_context(decoder, error)) {
        return context_failure(error);
    }
    CUVIDSOURCEDATAPACKET packet{};
    packet.flags = CUVID_PKT_TIMESTAMP;
    packet.payload_size = data_len;
    packet.payload = data;
    packet.timestamp = timestamp_100ns;
    decoder->callback_error = CUDA_SUCCESS;
    const CUresult status = decoder->cuvid->cuvidParseVideoData(decoder->parser, &packet);
    pop_context(decoder);
    const CUresult effective = status == CUDA_SUCCESS ? decoder->callback_error : status;
    return effective == CUDA_SUCCESS ? 0 : cuda_fail(decoder, error, effective, "submit NVDEC packet");
#endif
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvdecoder_flush(
    NpNvDecoder *decoder,
    NpVideoError *error) noexcept {
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    (void)decoder;
    return invalid_probe(error);
#else
    if (decoder == nullptr || !push_context(decoder, error)) {
        return context_failure(error);
    }
    CUVIDSOURCEDATAPACKET packet{};
    packet.flags = CUVID_PKT_ENDOFSTREAM;
    decoder->callback_error = CUDA_SUCCESS;
    const CUresult status = decoder->cuvid->cuvidParseVideoData(decoder->parser, &packet);
    pop_context(decoder);
    const CUresult effective = status == CUDA_SUCCESS ? decoder->callback_error : status;
    return effective == CUDA_SUCCESS ? 0 : cuda_fail(decoder, error, effective, "flush NVDEC parser");
#endif
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvdecoder_map(
    NpNvDecoder *decoder,
    NpCudaVideoSurface *out_surface,
    NpVideoError *error) noexcept {
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    (void)decoder; (void)out_surface;
    return invalid_probe(error);
#else
    if (decoder == nullptr || out_surface == nullptr) {
        return invalid_probe(error);
    }
    std::memset(out_surface, 0, sizeof(*out_surface));
    if (decoder->ready.empty()) {
        return 0;
    }
    if (!push_context(decoder, error)) {
        return context_failure(error);
    }
    const CUVIDPARSERDISPINFO display = decoder->ready.front();
    CUVIDPROCPARAMS process{};
    process.progressive_frame = display.progressive_frame;
    process.top_field_first = display.top_field_first;
    process.unpaired_field = display.repeat_first_field < 0;
    process.output_stream = decoder->stream;
    CUdeviceptr pointer = 0;
    unsigned int pitch = 0;
    const CUresult status = decoder->cuvid->cuvidMapVideoFrame(
        decoder->decoder, display.picture_index, &pointer, &pitch, &process);
    pop_context(decoder);
    if (status != CUDA_SUCCESS) {
        return cuda_fail(decoder, error, status, "map NVDEC output surface");
    }
    try {
        decoder->mapped.push_back(pointer);
    } catch (...) {
        NpVideoError ignored{};
        push_context(decoder, &ignored);
        decoder->cuvid->cuvidUnmapVideoFrame(decoder->decoder, pointer);
        pop_context(decoder);
        return cuda_fail(decoder, error, CUDA_ERROR_UNKNOWN, "track NVDEC output surface");
    }
    decoder->ready.pop_front();
    out_surface->abi_version = NP_VIDEO_ABI_VERSION;
    out_surface->pixel_format = decoder->pixel_format;
    out_surface->device_ptr = pointer;
    out_surface->pitch = pitch;
    out_surface->width = decoder->width;
    out_surface->height = decoder->height;
    out_surface->timestamp_100ns = display.timestamp;
    out_surface->picture_index = display.picture_index;
    out_surface->progressive = display.progressive_frame != 0;
    return 1;
#endif
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvdecoder_unmap(
    NpNvDecoder *decoder,
    uint64_t device_ptr,
    NpVideoError *error) noexcept {
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    (void)decoder; (void)device_ptr;
    return invalid_probe(error);
#else
    if (decoder == nullptr || device_ptr == 0) {
        return invalid_probe(error);
    }
    const auto found = std::find(decoder->mapped.begin(), decoder->mapped.end(), device_ptr);
    if (found == decoder->mapped.end()) {
        if (error != nullptr) {
            error->code = -EINVAL;
            std::snprintf(error->message, sizeof(error->message), "NVDEC surface is not mapped");
        }
        return -EINVAL;
    }
    if (!push_context(decoder, error)) {
        return context_failure(error);
    }
    const CUresult status = decoder->cuvid->cuvidUnmapVideoFrame(decoder->decoder, device_ptr);
    pop_context(decoder);
    if (status != CUDA_SUCCESS) {
        return cuda_fail(decoder, error, status, "unmap NVDEC output surface");
    }
    decoder->mapped.erase(found);
    return 0;
#endif
}

extern "C" NP_VIDEO_EXPORT void np_video_nvdecoder_close(NpNvDecoder *decoder) noexcept {
    destroy_decoder(decoder);
}
