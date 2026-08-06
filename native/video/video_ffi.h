#ifndef NOPERSON_VIDEO_FFI_H
#define NOPERSON_VIDEO_FFI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define NP_VIDEO_EXPORT __declspec(dllexport)
#else
#define NP_VIDEO_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define NP_VIDEO_NOEXCEPT noexcept
#else
#define NP_VIDEO_NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { NP_VIDEO_ABI_VERSION = 1, NP_VIDEO_ERROR_CAPACITY = 1024 };

typedef struct NpVideoError {
    int32_t code;
    char message[NP_VIDEO_ERROR_CAPACITY];
} NpVideoError;

typedef struct NpVideoStreamInfo {
    uint32_t abi_version;
    int32_t index;
    int32_t codec;
    uint32_t width;
    uint32_t height;
    uint32_t bit_depth;
    int32_t time_base_num;
    int32_t time_base_den;
    uint32_t frame_rate_num;
    uint32_t frame_rate_den;
    int64_t frame_count;
    int64_t duration_ts;
    int32_t color_range;
    int32_t color_matrix;
    int32_t color_primaries;
    int32_t color_transfer;
    int32_t chroma_location;
    const uint8_t *extradata;
    size_t extradata_len;
} NpVideoStreamInfo;

typedef struct NpVideoPacket {
    const uint8_t *data;
    size_t data_len;
    int64_t pts;
    int64_t dts;
    int64_t duration;
    int32_t stream_index;
    uint32_t flags;
} NpVideoPacket;

typedef struct NpVideoDemuxer NpVideoDemuxer;
typedef struct NpVideoMuxer NpVideoMuxer;

typedef struct NpNvCodecCapabilities {
    uint32_t abi_version;
    uint8_t cuda_driver;
    uint8_t nvdec;
    uint8_t nvenc;
    uint8_t reserved;
    uint32_t nvenc_api_major;
    uint32_t nvenc_api_minor;
} NpNvCodecCapabilities;

typedef struct NpCudaVideoSurface {
    uint32_t abi_version;
    uint32_t pixel_format;
    uint64_t device_ptr;
    uint32_t pitch;
    uint32_t width;
    uint32_t height;
    int64_t timestamp_100ns;
    int32_t picture_index;
    uint32_t progressive;
} NpCudaVideoSurface;

typedef struct NpNvDecoder NpNvDecoder;
typedef struct NpNvEncoder NpNvEncoder;

typedef struct NpNvEncoderConfig {
    uint32_t abi_version;
    int32_t codec;
    uint32_t pixel_format;
    uint32_t width;
    uint32_t height;
    uint32_t frame_rate_num;
    uint32_t frame_rate_den;
    int32_t time_base_num;
    int32_t time_base_den;
    uint32_t constant_qp;
    uint32_t gop_length;
    int32_t color_range;
    int32_t color_matrix;
    int32_t color_primaries;
    int32_t color_transfer;
    int32_t chroma_location;
} NpNvEncoderConfig;

NP_VIDEO_EXPORT int32_t np_video_nvcodec_probe(
    NpNvCodecCapabilities *out_capabilities,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_nvdecoder_open(
    void *cuda_context,
    void *cuda_stream,
    int32_t codec,
    NpNvDecoder **out_decoder,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_nvdecoder_send(
    NpNvDecoder *decoder,
    const uint8_t *data,
    size_t data_len,
    int64_t timestamp_100ns,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_nvdecoder_flush(
    NpNvDecoder *decoder,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

// Returns 1 and the next decoded picture slot, or 0 when no frame is ready.
NP_VIDEO_EXPORT int32_t np_video_nvdecoder_peek(
    NpNvDecoder *decoder,
    int32_t *out_picture_index,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

// Returns 1 for a mapped surface, 0 when no decoded frame is ready, negative on error.
NP_VIDEO_EXPORT int32_t np_video_nvdecoder_map(
    NpNvDecoder *decoder,
    NpCudaVideoSurface *out_surface,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_nvdecoder_unmap(
    NpNvDecoder *decoder,
    uint64_t device_ptr,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT void np_video_nvdecoder_close(NpNvDecoder *decoder) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_nvencoder_open(
    void *cuda_context,
    void *cuda_stream,
    const NpNvEncoderConfig *config,
    NpNvEncoder **out_encoder,
    NpVideoStreamInfo *out_video,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

// Returns 1 for an encoded packet, 0 when NVENC buffered the input, negative on error.
NP_VIDEO_EXPORT int32_t np_video_nvencoder_encode(
    NpNvEncoder *encoder,
    uint64_t device_ptr,
    uint32_t pitch,
    int64_t pts,
    int64_t duration,
    NpVideoPacket *out_packet,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_nvencoder_flush(
    NpNvEncoder *encoder,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

// Returns 1 for a completed packet, 0 when the ring is empty, negative on error.
NP_VIDEO_EXPORT int32_t np_video_nvencoder_receive(
    NpNvEncoder *encoder,
    NpVideoPacket *out_packet,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT void np_video_nvencoder_close(NpNvEncoder *encoder) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_demux_open(
    const char *path,
    NpVideoDemuxer **out_demuxer,
    NpVideoStreamInfo *out_video,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

// Returns 1 for a packet, 0 for end-of-file, and a negative value on error.
NP_VIDEO_EXPORT int32_t np_video_demux_read(
    NpVideoDemuxer *demuxer,
    NpVideoPacket *out_packet,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

// Reads a video packet normalized to the elementary bitstream expected by NVDEC.
NP_VIDEO_EXPORT int32_t np_video_demux_read_decode(
    NpVideoDemuxer *demuxer,
    NpVideoPacket *out_packet,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT void np_video_demux_close(NpVideoDemuxer *demuxer) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_mux_open(
    const char *path,
    const NpVideoStreamInfo *video,
    NpVideoMuxer **out_muxer,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

NP_VIDEO_EXPORT int32_t np_video_mux_write(
    NpVideoMuxer *muxer,
    const NpVideoPacket *packet,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

// Writes the trailer and destroys the muxer, including on failure.
NP_VIDEO_EXPORT int32_t np_video_mux_finish(
    NpVideoMuxer *muxer,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

// Aborts output without writing a trailer.
NP_VIDEO_EXPORT void np_video_mux_close(NpVideoMuxer *muxer) NP_VIDEO_NOEXCEPT;

// Replaces the source video with the processed stream while copying every
// source audio stream in-process through libavformat.
NP_VIDEO_EXPORT int32_t np_video_remux_audio(
    const char *video_path,
    const char *source_path,
    const char *output_path,
    NpVideoError *error) NP_VIDEO_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#endif
