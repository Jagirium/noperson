#include "video_ffi.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#if defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
#define FFNV_LOG_FUNC(logctx, msg, ...) ((void)(logctx))
#define FFNV_DEBUG_LOG_FUNC(logctx, msg, ...) ((void)(logctx))
#include <ffnvcodec/dynlink_loader.h>
#endif

struct NpNvEncoder {
#if defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    struct PendingEncode {
        NV_ENC_OUTPUT_PTR bitstream = nullptr;
        NV_ENC_INPUT_PTR mapped = nullptr;
    };
    NvencFunctions *loader = nullptr;
    NV_ENCODE_API_FUNCTION_LIST api{};
    void *session = nullptr;
    void *stream = nullptr;
    std::vector<NV_ENC_OUTPUT_PTR> bitstreams;
    std::vector<NV_ENC_OUTPUT_PTR> free_bitstreams;
    std::vector<PendingEncode> pending;
    std::vector<std::pair<uint64_t, NV_ENC_REGISTERED_PTR>> resources;
    std::vector<uint8_t> packet;
    std::vector<uint8_t> sequence;
    NpNvEncoderConfig config{};
#endif
};

namespace {

void clear_error(NpVideoError *error) noexcept {
    if (error != nullptr) {
        error->code = 0;
        error->message[0] = '\0';
    }
}

int32_t generic_error(NpVideoError *error, int32_t code, const char *message) noexcept {
    if (error != nullptr) {
        error->code = code;
        std::snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return code < 0 ? code : -code;
}

#if defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
int32_t nvenc_error(NpNvEncoder *encoder, NpVideoError *error, NVENCSTATUS status,
                    const char *operation) noexcept {
    const char *detail = nullptr;
    if (encoder != nullptr && encoder->session != nullptr &&
        encoder->api.nvEncGetLastErrorString != nullptr) {
        detail = encoder->api.nvEncGetLastErrorString(encoder->session);
    }
    if (error != nullptr) {
        error->code = static_cast<int32_t>(status);
        std::snprintf(error->message, sizeof(error->message), "%s: %s (NVENC %d)",
                      operation, detail != nullptr ? detail : "NVENC failure",
                      static_cast<int>(status));
    }
    return -static_cast<int32_t>(status == NV_ENC_SUCCESS ? NV_ENC_ERR_GENERIC : status);
}

GUID encode_guid(int32_t codec) noexcept {
    return codec == 2 ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;
}

NV_ENC_BUFFER_FORMAT buffer_format(uint32_t format) noexcept {
    return format == 2 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12;
}

void destroy_encoder(NpNvEncoder *encoder) noexcept {
    if (encoder == nullptr) {
        return;
    }
    if (encoder->session != nullptr) {
        for (const auto &pending : encoder->pending) {
            if (pending.mapped != nullptr) {
                encoder->api.nvEncUnmapInputResource(encoder->session, pending.mapped);
            }
        }
        for (const auto &resource : encoder->resources) {
            encoder->api.nvEncUnregisterResource(encoder->session, resource.second);
        }
        for (const auto bitstream : encoder->bitstreams) {
            encoder->api.nvEncDestroyBitstreamBuffer(encoder->session, bitstream);
        }
        encoder->api.nvEncDestroyEncoder(encoder->session);
    }
    nvenc_free_functions(&encoder->loader);
    delete encoder;
}

NV_ENC_REGISTERED_PTR register_resource(NpNvEncoder *encoder, uint64_t pointer,
                                        uint32_t pitch, NpVideoError *error) noexcept {
    for (const auto &resource : encoder->resources) {
        if (resource.first == pointer) {
            return resource.second;
        }
    }
    NV_ENC_REGISTER_RESOURCE registration{};
    registration.version = NV_ENC_REGISTER_RESOURCE_VER;
    registration.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    registration.width = encoder->config.width;
    registration.height = encoder->config.height;
    registration.pitch = pitch;
    registration.resourceToRegister = reinterpret_cast<void *>(pointer);
    registration.bufferFormat = buffer_format(encoder->config.pixel_format);
    registration.bufferUsage = NV_ENC_INPUT_IMAGE;
    const NVENCSTATUS status = encoder->api.nvEncRegisterResource(encoder->session, &registration);
    if (status != NV_ENC_SUCCESS) {
        nvenc_error(encoder, error, status, "register CUDA frame with NVENC");
        return nullptr;
    }
    try {
        encoder->resources.emplace_back(pointer, registration.registeredResource);
    } catch (...) {
        encoder->api.nvEncUnregisterResource(encoder->session, registration.registeredResource);
        generic_error(error, -ENOMEM, "track registered NVENC resource failed");
        return nullptr;
    }
    return registration.registeredResource;
}

int32_t drain_packet(NpNvEncoder *encoder, NpVideoPacket *out_packet,
                     NpVideoError *error) noexcept {
    if (encoder->pending.empty()) {
        return 0;
    }
    const NpNvEncoder::PendingEncode pending = encoder->pending.front();
    NV_ENC_LOCK_BITSTREAM lock{};
    lock.version = NV_ENC_LOCK_BITSTREAM_VER;
    lock.outputBitstream = pending.bitstream;
    lock.doNotWait = 0;
    const NVENCSTATUS status = encoder->api.nvEncLockBitstream(encoder->session, &lock);
    if (status != NV_ENC_SUCCESS) {
        return nvenc_error(encoder, error, status, "lock NVENC bitstream");
    }
    try {
        const auto *begin = static_cast<const uint8_t *>(lock.bitstreamBufferPtr);
        encoder->packet.assign(begin, begin + lock.bitstreamSizeInBytes);
    } catch (...) {
        encoder->api.nvEncUnlockBitstream(encoder->session, pending.bitstream);
        return generic_error(error, -ENOMEM, "copy NVENC packet failed");
    }
    encoder->api.nvEncUnlockBitstream(encoder->session, pending.bitstream);
    encoder->api.nvEncUnmapInputResource(encoder->session, pending.mapped);
    encoder->pending.erase(encoder->pending.begin());
    encoder->free_bitstreams.push_back(pending.bitstream);
    out_packet->data = encoder->packet.data();
    out_packet->data_len = encoder->packet.size();
    out_packet->pts = static_cast<int64_t>(lock.outputTimeStamp);
    out_packet->dts = static_cast<int64_t>(lock.outputTimeStamp);
    out_packet->duration = static_cast<int64_t>(lock.outputDuration);
    out_packet->stream_index = 0;
    out_packet->flags = (lock.pictureType == NV_ENC_PIC_TYPE_IDR ||
                         lock.pictureType == NV_ENC_PIC_TYPE_I)
                            ? 1U
                            : 0U;
    return 1;
}
#else
void destroy_encoder(NpNvEncoder *encoder) noexcept { delete encoder; }
#endif

} // namespace

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvencoder_open(
    void *cuda_context,
    void *cuda_stream,
    const NpNvEncoderConfig *config,
    NpNvEncoder **out_encoder,
    NpVideoStreamInfo *out_video,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (cuda_context == nullptr || config == nullptr || out_encoder == nullptr ||
        out_video == nullptr || config->abi_version != NP_VIDEO_ABI_VERSION ||
        config->width == 0 || config->height == 0 || config->frame_rate_num == 0 ||
        config->frame_rate_den == 0 || config->time_base_num <= 0 ||
        config->time_base_den <= 0 || (config->codec != 1 && config->codec != 2) ||
        (config->pixel_format != 1 && config->pixel_format != 2)) {
        return generic_error(error, -EINVAL, "invalid NVENC configuration");
    }
    *out_encoder = nullptr;
    std::memset(out_video, 0, sizeof(*out_video));
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    (void)cuda_stream;
    return generic_error(error, -ENOSYS, "NVCodec headers were unavailable at build time");
#else
    if (config->codec == 1 && config->pixel_format == 2) {
        return generic_error(error, -EINVAL, "H.264 NVENC does not support P010 input");
    }
    auto *encoder = new (std::nothrow) NpNvEncoder();
    if (encoder == nullptr) {
        return generic_error(error, -ENOMEM, "allocate NVENC session failed");
    }
    encoder->config = *config;
    encoder->stream = cuda_stream;
    if (nvenc_load_functions(&encoder->loader, nullptr) != 0) {
        destroy_encoder(encoder);
        return generic_error(error, -ENOENT, "load NVENC driver API failed");
    }
    encoder->api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NVENCSTATUS status = encoder->loader->NvEncodeAPICreateInstance(&encoder->api);
    if (status != NV_ENC_SUCCESS) {
        const int32_t result = nvenc_error(encoder, error, status, "create NVENC API instance");
        destroy_encoder(encoder);
        return result;
    }
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session{};
    session.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    session.device = cuda_context;
    session.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    session.apiVersion = NVENCAPI_VERSION;
    status = encoder->api.nvEncOpenEncodeSessionEx(&session, &encoder->session);
    if (status != NV_ENC_SUCCESS) {
        const int32_t result = nvenc_error(encoder, error, status, "open NVENC session");
        destroy_encoder(encoder);
        return result;
    }

    const GUID codec = encode_guid(config->codec);
    NV_ENC_PRESET_CONFIG preset{};
    preset.version = NV_ENC_PRESET_CONFIG_VER;
    preset.presetCfg.version = NV_ENC_CONFIG_VER;
    status = encoder->api.nvEncGetEncodePresetConfigEx(
        encoder->session, codec, NV_ENC_PRESET_P7_GUID,
        NV_ENC_TUNING_INFO_HIGH_QUALITY, &preset);
    if (status != NV_ENC_SUCCESS) {
        const int32_t result = nvenc_error(encoder, error, status, "load NVENC P7 preset");
        destroy_encoder(encoder);
        return result;
    }
    NV_ENC_CONFIG encode = preset.presetCfg;
    encode.version = NV_ENC_CONFIG_VER;
    encode.gopLength = config->gop_length;
    encode.frameIntervalP = 1;
    encode.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    encode.rcParams.constQP.qpInterP = config->constant_qp;
    encode.rcParams.constQP.qpInterB = config->constant_qp;
    encode.rcParams.constQP.qpIntra = config->constant_qp;
    encode.rcParams.enableLookahead = 0;
    if (config->codec == 1) {
        encode.encodeCodecConfig.h264Config.idrPeriod = config->gop_length;
        encode.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
        auto &vui = encode.encodeCodecConfig.h264Config.h264VUIParameters;
        vui.videoSignalTypePresentFlag = 1;
        vui.videoFullRangeFlag = config->color_range == 2;
        vui.colourDescriptionPresentFlag = 1;
        vui.colourPrimaries = static_cast<NV_ENC_VUI_COLOR_PRIMARIES>(config->color_primaries);
        vui.transferCharacteristics =
            static_cast<NV_ENC_VUI_TRANSFER_CHARACTERISTIC>(config->color_transfer);
        vui.colourMatrix = static_cast<NV_ENC_VUI_MATRIX_COEFFS>(config->color_matrix);
    } else {
        encode.encodeCodecConfig.hevcConfig.idrPeriod = config->gop_length;
        encode.encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;
        auto &vui = encode.encodeCodecConfig.hevcConfig.hevcVUIParameters;
        vui.videoSignalTypePresentFlag = 1;
        vui.videoFullRangeFlag = config->color_range == 2;
        vui.colourDescriptionPresentFlag = 1;
        vui.colourPrimaries = static_cast<NV_ENC_VUI_COLOR_PRIMARIES>(config->color_primaries);
        vui.transferCharacteristics =
            static_cast<NV_ENC_VUI_TRANSFER_CHARACTERISTIC>(config->color_transfer);
        vui.colourMatrix = static_cast<NV_ENC_VUI_MATRIX_COEFFS>(config->color_matrix);
    }

    NV_ENC_INITIALIZE_PARAMS initialize{};
    initialize.version = NV_ENC_INITIALIZE_PARAMS_VER;
    initialize.encodeGUID = codec;
    initialize.presetGUID = NV_ENC_PRESET_P7_GUID;
    initialize.encodeWidth = config->width;
    initialize.encodeHeight = config->height;
    initialize.darWidth = config->width;
    initialize.darHeight = config->height;
    initialize.frameRateNum = config->frame_rate_num;
    initialize.frameRateDen = config->frame_rate_den;
    initialize.enablePTD = 1;
    initialize.enableEncodeAsync = 0;
    initialize.maxEncodeWidth = config->width;
    initialize.maxEncodeHeight = config->height;
    initialize.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    initialize.encodeConfig = &encode;
    status = encoder->api.nvEncInitializeEncoder(encoder->session, &initialize);
    if (status != NV_ENC_SUCCESS) {
        const int32_t result = nvenc_error(encoder, error, status, "initialize NVENC encoder");
        destroy_encoder(encoder);
        return result;
    }
    if (encoder->api.nvEncSetIOCudaStreams != nullptr && cuda_stream != nullptr) {
        status = encoder->api.nvEncSetIOCudaStreams(encoder->session, cuda_stream, cuda_stream);
        if (status != NV_ENC_SUCCESS) {
            const int32_t result = nvenc_error(encoder, error, status, "bind NVENC CUDA stream");
            destroy_encoder(encoder);
            return result;
        }
    }
    constexpr size_t bitstream_ring_size = 4;
    try {
        encoder->bitstreams.reserve(bitstream_ring_size);
        encoder->free_bitstreams.reserve(bitstream_ring_size);
        encoder->pending.reserve(bitstream_ring_size);
    } catch (...) {
        destroy_encoder(encoder);
        return generic_error(error, -ENOMEM, "allocate NVENC bitstream ring failed");
    }
    for (size_t index = 0; index < bitstream_ring_size; ++index) {
        NV_ENC_CREATE_BITSTREAM_BUFFER bitstream{};
        bitstream.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        status = encoder->api.nvEncCreateBitstreamBuffer(encoder->session, &bitstream);
        if (status != NV_ENC_SUCCESS) {
            const int32_t result =
                nvenc_error(encoder, error, status, "create NVENC bitstream ring");
            destroy_encoder(encoder);
            return result;
        }
        encoder->bitstreams.push_back(bitstream.bitstreamBuffer);
        encoder->free_bitstreams.push_back(bitstream.bitstreamBuffer);
    }

    uint8_t sequence[4096]{};
    uint32_t sequence_size = 0;
    NV_ENC_SEQUENCE_PARAM_PAYLOAD payload{};
    payload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
    payload.spsppsBuffer = sequence;
    payload.inBufferSize = sizeof(sequence);
    payload.outSPSPPSPayloadSize = &sequence_size;
    status = encoder->api.nvEncGetSequenceParams(encoder->session, &payload);
    if (status == NV_ENC_SUCCESS) {
        try {
            encoder->sequence.assign(sequence, sequence + sequence_size);
        } catch (...) {
            destroy_encoder(encoder);
            return generic_error(error, -ENOMEM, "copy NVENC sequence parameters failed");
        }
    }
    out_video->abi_version = NP_VIDEO_ABI_VERSION;
    out_video->index = 0;
    out_video->codec = config->codec;
    out_video->width = config->width;
    out_video->height = config->height;
    out_video->bit_depth = config->pixel_format == 2 ? 10U : 8U;
    out_video->time_base_num = config->time_base_num;
    out_video->time_base_den = config->time_base_den;
    out_video->frame_rate_num = config->frame_rate_num;
    out_video->frame_rate_den = config->frame_rate_den;
    out_video->frame_count = 0;
    out_video->color_range = config->color_range;
    out_video->color_matrix = config->color_matrix;
    out_video->color_primaries = config->color_primaries;
    out_video->color_transfer = config->color_transfer;
    out_video->chroma_location = config->chroma_location;
    // libavformat accepts Annex-B SPS/PPS here and writes the container-specific
    // avcC/hvcC representation. NVENC also repeats them in every IDR.
    out_video->extradata = encoder->sequence.data();
    out_video->extradata_len = encoder->sequence.size();
    *out_encoder = encoder;
    return 0;
#endif
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvencoder_encode(
    NpNvEncoder *encoder,
    uint64_t device_ptr,
    uint32_t pitch,
    int64_t pts,
    int64_t duration,
    NpVideoPacket *out_packet,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (encoder == nullptr || device_ptr == 0 || pitch == 0 || out_packet == nullptr) {
        return generic_error(error, -EINVAL, "invalid NVENC frame");
    }
    std::memset(out_packet, 0, sizeof(*out_packet));
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    (void)pts;
    (void)duration;
    return generic_error(error, -ENOSYS, "NVCodec headers were unavailable at build time");
#else
    int32_t produced = 0;
    if (encoder->free_bitstreams.empty()) {
        produced = drain_packet(encoder, out_packet, error);
        if (produced < 0) {
            return produced;
        }
    }
    const NV_ENC_REGISTERED_PTR registered = register_resource(encoder, device_ptr, pitch, error);
    if (registered == nullptr) {
        return error != nullptr && error->code != 0
                   ? (error->code < 0 ? error->code : -error->code)
                   : -1;
    }
    NV_ENC_MAP_INPUT_RESOURCE mapping{};
    mapping.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    mapping.registeredResource = registered;
    NVENCSTATUS status = encoder->api.nvEncMapInputResource(encoder->session, &mapping);
    if (status != NV_ENC_SUCCESS) {
        return nvenc_error(encoder, error, status, "map NVENC input resource");
    }
    NV_ENC_PIC_PARAMS picture{};
    picture.version = NV_ENC_PIC_PARAMS_VER;
    picture.inputBuffer = mapping.mappedResource;
    picture.bufferFmt = buffer_format(encoder->config.pixel_format);
    picture.inputWidth = encoder->config.width;
    picture.inputHeight = encoder->config.height;
    const NV_ENC_OUTPUT_PTR bitstream = encoder->free_bitstreams.back();
    encoder->free_bitstreams.pop_back();
    picture.outputBitstream = bitstream;
    picture.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picture.inputTimeStamp = static_cast<uint64_t>(pts);
    picture.inputDuration = static_cast<uint64_t>(duration);
    status = encoder->api.nvEncEncodePicture(encoder->session, &picture);
    if (status != NV_ENC_SUCCESS && status != NV_ENC_ERR_NEED_MORE_INPUT) {
        encoder->api.nvEncUnmapInputResource(encoder->session, mapping.mappedResource);
        encoder->free_bitstreams.push_back(bitstream);
        return nvenc_error(encoder, error, status, "encode NVENC frame");
    }
    encoder->pending.push_back({bitstream, mapping.mappedResource});
    return produced;
#endif
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvencoder_flush(
    NpNvEncoder *encoder,
    NpVideoError *error) noexcept {
    clear_error(error);
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    (void)encoder;
    return generic_error(error, -ENOSYS, "NVCodec headers were unavailable at build time");
#else
    if (encoder == nullptr) {
        return generic_error(error, -EINVAL, "invalid NVENC flush argument");
    }
    NV_ENC_PIC_PARAMS picture{};
    picture.version = NV_ENC_PIC_PARAMS_VER;
    picture.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    const NVENCSTATUS status = encoder->api.nvEncEncodePicture(encoder->session, &picture);
    return status == NV_ENC_SUCCESS ? 0 : nvenc_error(encoder, error, status, "flush NVENC encoder");
#endif
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_nvencoder_receive(
    NpNvEncoder *encoder,
    NpVideoPacket *out_packet,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (encoder == nullptr || out_packet == nullptr) {
        return generic_error(error, -EINVAL, "invalid NVENC receive arguments");
    }
    std::memset(out_packet, 0, sizeof(*out_packet));
#if !defined(NP_VIDEO_HAS_NV_CODEC_HEADERS)
    return generic_error(error, -ENOSYS, "NVCodec headers were unavailable at build time");
#else
    return drain_packet(encoder, out_packet, error);
#endif
}

extern "C" NP_VIDEO_EXPORT void np_video_nvencoder_close(NpNvEncoder *encoder) noexcept {
    destroy_encoder(encoder);
}
