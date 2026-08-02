#include "video_ffi.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <new>

extern "C" {
#include <libavcodec/codec_id.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/pixdesc.h>
}

struct NpVideoDemuxer {
    AVFormatContext *format = nullptr;
    AVPacket *packet = nullptr;
    AVPacket *filtered = nullptr;
    AVBSFContext *bitstream_filter = nullptr;
    int video_stream = -1;
    bool filter_eof_sent = false;
};

namespace {

void clear_error(NpVideoError *error) noexcept {
    if (error == nullptr) {
        return;
    }
    error->code = 0;
    error->message[0] = '\0';
}

int32_t fail(NpVideoError *error, int32_t code, const char *context) noexcept {
    if (error != nullptr) {
        error->code = code;
        char detail[AV_ERROR_MAX_STRING_SIZE] = {};
        if (code < 0) {
            av_strerror(code, detail, sizeof(detail));
        } else {
            std::snprintf(detail, sizeof(detail), "error %d", code);
        }
        std::snprintf(error->message, sizeof(error->message), "%s: %s", context, detail);
    }
    return code < 0 ? code : -code;
}

int32_t codec_kind(AVCodecID codec) noexcept {
    switch (codec) {
    case AV_CODEC_ID_H264:
        return 1;
    case AV_CODEC_ID_HEVC:
        return 2;
    case AV_CODEC_ID_AV1:
        return 3;
    case AV_CODEC_ID_VP9:
        return 4;
    default:
        return 0;
    }
}

void destroy(NpVideoDemuxer *demuxer) noexcept {
    if (demuxer == nullptr) {
        return;
    }
    av_packet_free(&demuxer->packet);
    av_packet_free(&demuxer->filtered);
    av_bsf_free(&demuxer->bitstream_filter);
    avformat_close_input(&demuxer->format);
    delete demuxer;
}

} // namespace

extern "C" NP_VIDEO_EXPORT int32_t np_video_demux_open(
    const char *path,
    NpVideoDemuxer **out_demuxer,
    NpVideoStreamInfo *out_video,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (path == nullptr || out_demuxer == nullptr || out_video == nullptr) {
        return fail(error, AVERROR(EINVAL), "invalid demux open arguments");
    }
    *out_demuxer = nullptr;
    std::memset(out_video, 0, sizeof(*out_video));

    auto *demuxer = new (std::nothrow) NpVideoDemuxer();
    if (demuxer == nullptr) {
        return fail(error, AVERROR(ENOMEM), "allocate demuxer");
    }
    int status = avformat_open_input(&demuxer->format, path, nullptr, nullptr);
    if (status < 0) {
        destroy(demuxer);
        return fail(error, status, "open media input");
    }
    status = avformat_find_stream_info(demuxer->format, nullptr);
    if (status < 0) {
        destroy(demuxer);
        return fail(error, status, "read media stream information");
    }
    status = av_find_best_stream(
        demuxer->format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (status < 0) {
        destroy(demuxer);
        return fail(error, status, "find video stream");
    }
    demuxer->video_stream = status;
    demuxer->packet = av_packet_alloc();
    demuxer->filtered = av_packet_alloc();
    if (demuxer->packet == nullptr || demuxer->filtered == nullptr) {
        destroy(demuxer);
        return fail(error, AVERROR(ENOMEM), "allocate media packet");
    }

    AVStream *stream = demuxer->format->streams[demuxer->video_stream];
    const AVCodecParameters *codec = stream->codecpar;
    const int32_t kind = codec_kind(codec->codec_id);
    if (kind == 0) {
        destroy(demuxer);
        return fail(error, AVERROR_DECODER_NOT_FOUND, "unsupported video codec");
    }
    const char *filter_name = nullptr;
    if (codec->codec_id == AV_CODEC_ID_H264) {
        filter_name = "h264_mp4toannexb";
    } else if (codec->codec_id == AV_CODEC_ID_HEVC) {
        filter_name = "hevc_mp4toannexb";
    }
    if (filter_name != nullptr) {
        const AVBitStreamFilter *filter = av_bsf_get_by_name(filter_name);
        if (filter == nullptr) {
            destroy(demuxer);
            return fail(error, AVERROR_BSF_NOT_FOUND, "find NVDEC bitstream filter");
        }
        status = av_bsf_alloc(filter, &demuxer->bitstream_filter);
        if (status >= 0) {
            status = avcodec_parameters_copy(demuxer->bitstream_filter->par_in, codec);
        }
        if (status >= 0) {
            demuxer->bitstream_filter->time_base_in = stream->time_base;
            status = av_bsf_init(demuxer->bitstream_filter);
        }
        if (status < 0) {
            destroy(demuxer);
            return fail(error, status, "initialize NVDEC bitstream filter");
        }
    }
    out_video->abi_version = NP_VIDEO_ABI_VERSION;
    out_video->index = demuxer->video_stream;
    out_video->codec = kind;
    out_video->width = static_cast<uint32_t>(codec->width);
    out_video->height = static_cast<uint32_t>(codec->height);
    const AVPixFmtDescriptor *pixel = av_pix_fmt_desc_get(
        static_cast<AVPixelFormat>(codec->format));
    out_video->bit_depth = pixel != nullptr && pixel->nb_components > 0
                               ? static_cast<uint32_t>(pixel->comp[0].depth)
                               : static_cast<uint32_t>(codec->bits_per_raw_sample > 0
                                                           ? codec->bits_per_raw_sample
                                                           : 8);
    out_video->time_base_num = stream->time_base.num;
    out_video->time_base_den = stream->time_base.den;
    const AVRational frame_rate = av_guess_frame_rate(demuxer->format, stream, nullptr);
    out_video->frame_rate_num = frame_rate.num > 0 ? static_cast<uint32_t>(frame_rate.num) : 0;
    out_video->frame_rate_den = frame_rate.den > 0 ? static_cast<uint32_t>(frame_rate.den) : 0;
    out_video->frame_count = stream->nb_frames > 0 ? stream->nb_frames : 0;
    out_video->color_range = codec->color_range;
    out_video->color_matrix = codec->color_space;
    out_video->color_primaries = codec->color_primaries;
    out_video->color_transfer = codec->color_trc;
    out_video->chroma_location = codec->chroma_location;
    out_video->extradata = codec->extradata;
    out_video->extradata_len = static_cast<size_t>(codec->extradata_size);
    *out_demuxer = demuxer;
    return 0;
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_demux_read(
    NpVideoDemuxer *demuxer,
    NpVideoPacket *out_packet,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (demuxer == nullptr || out_packet == nullptr) {
        return fail(error, AVERROR(EINVAL), "invalid demux read arguments");
    }
    std::memset(out_packet, 0, sizeof(*out_packet));
    av_packet_unref(demuxer->packet);
    for (;;) {
        const int status = av_read_frame(demuxer->format, demuxer->packet);
        if (status == AVERROR_EOF) {
            return 0;
        }
        if (status < 0) {
            return fail(error, status, "read media packet");
        }
        if (demuxer->packet->stream_index != demuxer->video_stream) {
            av_packet_unref(demuxer->packet);
            continue;
        }
        out_packet->data = demuxer->packet->data;
        out_packet->data_len = static_cast<size_t>(demuxer->packet->size);
        out_packet->pts = demuxer->packet->pts;
        out_packet->dts = demuxer->packet->dts;
        out_packet->duration = demuxer->packet->duration;
        out_packet->stream_index = demuxer->packet->stream_index;
        out_packet->flags = static_cast<uint32_t>(demuxer->packet->flags);
        return 1;
    }
}

extern "C" NP_VIDEO_EXPORT void np_video_demux_close(NpVideoDemuxer *demuxer) noexcept {
    destroy(demuxer);
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_demux_read_decode(
    NpVideoDemuxer *demuxer,
    NpVideoPacket *out_packet,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (demuxer == nullptr || out_packet == nullptr) {
        return fail(error, AVERROR(EINVAL), "invalid decode packet arguments");
    }
    if (demuxer->bitstream_filter == nullptr) {
        return np_video_demux_read(demuxer, out_packet, error);
    }
    std::memset(out_packet, 0, sizeof(*out_packet));
    av_packet_unref(demuxer->filtered);
    for (;;) {
        int status = av_bsf_receive_packet(demuxer->bitstream_filter, demuxer->filtered);
        if (status == 0) {
            out_packet->data = demuxer->filtered->data;
            out_packet->data_len = static_cast<size_t>(demuxer->filtered->size);
            out_packet->pts = demuxer->filtered->pts;
            out_packet->dts = demuxer->filtered->dts;
            out_packet->duration = demuxer->filtered->duration;
            out_packet->stream_index = demuxer->video_stream;
            out_packet->flags = static_cast<uint32_t>(demuxer->filtered->flags);
            return 1;
        }
        if (status == AVERROR_EOF) {
            return 0;
        }
        if (status != AVERROR(EAGAIN)) {
            return fail(error, status, "filter NVDEC bitstream packet");
        }
        if (demuxer->filter_eof_sent) {
            return 0;
        }
        av_packet_unref(demuxer->packet);
        do {
            status = av_read_frame(demuxer->format, demuxer->packet);
        } while (status >= 0 && demuxer->packet->stream_index != demuxer->video_stream &&
                 (av_packet_unref(demuxer->packet), true));
        if (status == AVERROR_EOF) {
            demuxer->filter_eof_sent = true;
            status = av_bsf_send_packet(demuxer->bitstream_filter, nullptr);
        } else if (status >= 0) {
            status = av_bsf_send_packet(demuxer->bitstream_filter, demuxer->packet);
        }
        if (status < 0 && status != AVERROR_EOF) {
            return fail(error, status, "submit NVDEC bitstream packet");
        }
    }
}
