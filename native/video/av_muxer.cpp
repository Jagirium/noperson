#include "video_ffi.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

extern "C" {
#include <libavcodec/codec_id.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/mem.h>
}

struct NpVideoMuxer {
    AVFormatContext *format = nullptr;
    AVStream *video = nullptr;
    AVRational input_time_base{};
};

namespace {

void clear_error(NpVideoError *error) noexcept {
    if (error != nullptr) {
        error->code = 0;
        error->message[0] = '\0';
    }
}

int32_t fail(NpVideoError *error, int32_t code, const char *context) noexcept {
    if (error != nullptr) {
        error->code = code;
        char detail[AV_ERROR_MAX_STRING_SIZE] = {};
        av_strerror(code, detail, sizeof(detail));
        std::snprintf(error->message, sizeof(error->message), "%s: %s", context, detail);
    }
    return code < 0 ? code : -code;
}

AVCodecID codec_id(int32_t codec) noexcept {
    switch (codec) {
    case 1:
        return AV_CODEC_ID_H264;
    case 2:
        return AV_CODEC_ID_HEVC;
    case 3:
        return AV_CODEC_ID_AV1;
    case 4:
        return AV_CODEC_ID_VP9;
    default:
        return AV_CODEC_ID_NONE;
    }
}

void destroy(NpVideoMuxer *muxer) noexcept {
    if (muxer == nullptr) {
        return;
    }
    if (muxer->format != nullptr && muxer->format->pb != nullptr &&
        (muxer->format->oformat->flags & AVFMT_NOFILE) == 0) {
        avio_closep(&muxer->format->pb);
    }
    avformat_free_context(muxer->format);
    delete muxer;
}

} // namespace

extern "C" NP_VIDEO_EXPORT int32_t np_video_mux_open(
    const char *path,
    const NpVideoStreamInfo *video,
    NpVideoMuxer **out_muxer,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (path == nullptr || video == nullptr || out_muxer == nullptr) {
        return fail(error, AVERROR(EINVAL), "invalid mux open arguments");
    }
    *out_muxer = nullptr;
    if (video->abi_version != NP_VIDEO_ABI_VERSION || video->time_base_num <= 0 ||
        video->time_base_den <= 0 || video->width == 0 || video->height == 0) {
        return fail(error, AVERROR(EINVAL), "invalid mux video stream descriptor");
    }
    const AVCodecID id = codec_id(video->codec);
    if (id == AV_CODEC_ID_NONE) {
        return fail(error, AVERROR_ENCODER_NOT_FOUND, "unsupported mux video codec");
    }

    auto *muxer = new (std::nothrow) NpVideoMuxer();
    if (muxer == nullptr) {
        return fail(error, AVERROR(ENOMEM), "allocate muxer");
    }
    int status = avformat_alloc_output_context2(&muxer->format, nullptr, nullptr, path);
    if (status < 0 || muxer->format == nullptr) {
        destroy(muxer);
        return fail(error, status < 0 ? status : AVERROR(EINVAL), "create output container");
    }
    muxer->video = avformat_new_stream(muxer->format, nullptr);
    if (muxer->video == nullptr) {
        destroy(muxer);
        return fail(error, AVERROR(ENOMEM), "create output video stream");
    }
    muxer->video->time_base = AVRational{video->time_base_num, video->time_base_den};
    muxer->input_time_base = muxer->video->time_base;
    AVCodecParameters *parameters = muxer->video->codecpar;
    parameters->codec_type = AVMEDIA_TYPE_VIDEO;
    parameters->codec_id = id;
    parameters->codec_tag = 0;
    parameters->width = static_cast<int>(video->width);
    parameters->height = static_cast<int>(video->height);
    parameters->color_range = static_cast<AVColorRange>(video->color_range);
    parameters->color_space = static_cast<AVColorSpace>(video->color_matrix);
    parameters->color_primaries = static_cast<AVColorPrimaries>(video->color_primaries);
    parameters->color_trc = static_cast<AVColorTransferCharacteristic>(video->color_transfer);
    parameters->chroma_location = static_cast<AVChromaLocation>(video->chroma_location);
    if (video->extradata_len > 0) {
        if (video->extradata == nullptr || video->extradata_len > static_cast<size_t>(INT_MAX)) {
            destroy(muxer);
            return fail(error, AVERROR(EINVAL), "invalid codec extradata");
        }
        parameters->extradata = static_cast<uint8_t *>(
            av_mallocz(video->extradata_len + AV_INPUT_BUFFER_PADDING_SIZE));
        if (parameters->extradata == nullptr) {
            destroy(muxer);
            return fail(error, AVERROR(ENOMEM), "allocate codec extradata");
        }
        std::memcpy(parameters->extradata, video->extradata, video->extradata_len);
        parameters->extradata_size = static_cast<int>(video->extradata_len);
    }
    if ((muxer->format->oformat->flags & AVFMT_NOFILE) == 0) {
        status = avio_open(&muxer->format->pb, path, AVIO_FLAG_WRITE);
        if (status < 0) {
            destroy(muxer);
            return fail(error, status, "open output media file");
        }
    }
    status = avformat_write_header(muxer->format, nullptr);
    if (status < 0) {
        destroy(muxer);
        return fail(error, status, "write output media header");
    }
    *out_muxer = muxer;
    return 0;
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_mux_write(
    NpVideoMuxer *muxer,
    const NpVideoPacket *packet,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (muxer == nullptr || packet == nullptr ||
        (packet->data_len > 0 && packet->data == nullptr) ||
        packet->data_len > static_cast<size_t>(INT_MAX)) {
        return fail(error, AVERROR(EINVAL), "invalid mux packet");
    }
    AVPacket *native = av_packet_alloc();
    if (native == nullptr) {
        return fail(error, AVERROR(ENOMEM), "allocate output packet");
    }
    int status = av_new_packet(native, static_cast<int>(packet->data_len));
    if (status >= 0 && packet->data_len > 0) {
        std::memcpy(native->data, packet->data, packet->data_len);
    }
    if (status >= 0) {
        native->pts = packet->pts;
        native->dts = packet->dts;
        native->duration = packet->duration;
        native->stream_index = muxer->video->index;
        native->flags = (packet->flags & AV_PKT_FLAG_KEY) != 0 ? AV_PKT_FLAG_KEY : 0;
        av_packet_rescale_ts(native, muxer->input_time_base, muxer->video->time_base);
        status = av_interleaved_write_frame(muxer->format, native);
    }
    av_packet_free(&native);
    return status < 0 ? fail(error, status, "write output media packet") : 0;
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_mux_finish(
    NpVideoMuxer *muxer,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (muxer == nullptr) {
        return fail(error, AVERROR(EINVAL), "invalid mux finish argument");
    }
    const int status = av_write_trailer(muxer->format);
    destroy(muxer);
    return status < 0 ? fail(error, status, "write output media trailer") : 0;
}

extern "C" NP_VIDEO_EXPORT void np_video_mux_close(NpVideoMuxer *muxer) noexcept {
    destroy(muxer);
}

extern "C" NP_VIDEO_EXPORT int32_t np_video_remux_audio(
    const char *video_path,
    const char *source_path,
    const char *output_path,
    NpVideoError *error) noexcept {
    clear_error(error);
    if (video_path == nullptr || source_path == nullptr || output_path == nullptr) {
        return fail(error, AVERROR(EINVAL), "invalid audio remux arguments");
    }
    AVFormatContext *video = nullptr;
    AVFormatContext *source = nullptr;
    AVFormatContext *output = nullptr;
    AVPacket *video_packet = nullptr;
    AVPacket *audio_packet = nullptr;
    bool header_written = false;
    auto cleanup = [&]() noexcept {
        av_packet_free(&video_packet);
        av_packet_free(&audio_packet);
        if (output != nullptr && header_written) {
            av_write_trailer(output);
        }
        if (output != nullptr && output->pb != nullptr &&
            (output->oformat->flags & AVFMT_NOFILE) == 0) {
            avio_closep(&output->pb);
        }
        avformat_free_context(output);
        avformat_close_input(&source);
        avformat_close_input(&video);
    };
    int status = avformat_open_input(&video, video_path, nullptr, nullptr);
    if (status >= 0) {
        status = avformat_find_stream_info(video, nullptr);
    }
    if (status < 0) {
        cleanup();
        return fail(error, status, "open processed video for remux");
    }
    status = avformat_open_input(&source, source_path, nullptr, nullptr);
    if (status >= 0) {
        status = avformat_find_stream_info(source, nullptr);
    }
    if (status < 0) {
        cleanup();
        return fail(error, status, "open source audio for remux");
    }
    const int video_index =
        av_find_best_stream(video, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_index < 0) {
        cleanup();
        return fail(error, video_index, "find processed video stream");
    }
    status = avformat_alloc_output_context2(&output, nullptr, nullptr, output_path);
    if (status < 0 || output == nullptr) {
        cleanup();
        return fail(error, status < 0 ? status : AVERROR(EINVAL), "create remux output");
    }
    AVStream *output_video = avformat_new_stream(output, nullptr);
    if (output_video == nullptr) {
        cleanup();
        return fail(error, AVERROR(ENOMEM), "create remux video stream");
    }
    const AVStream *input_video = video->streams[video_index];
    status = avcodec_parameters_copy(output_video->codecpar, input_video->codecpar);
    if (status < 0) {
        cleanup();
        return fail(error, status, "copy remux video parameters");
    }
    output_video->codecpar->codec_tag = 0;
    output_video->time_base = input_video->time_base;

    std::vector<int> audio_mapping(source->nb_streams, -1);
    for (unsigned int index = 0; index < source->nb_streams; ++index) {
        const AVStream *input = source->streams[index];
        if (input->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) {
            continue;
        }
        AVStream *copy = avformat_new_stream(output, nullptr);
        if (copy == nullptr) {
            cleanup();
            return fail(error, AVERROR(ENOMEM), "create remux audio stream");
        }
        status = avcodec_parameters_copy(copy->codecpar, input->codecpar);
        if (status < 0) {
            cleanup();
            return fail(error, status, "copy remux audio parameters");
        }
        copy->codecpar->codec_tag = 0;
        copy->time_base = input->time_base;
        audio_mapping[index] = copy->index;
    }
    if ((output->oformat->flags & AVFMT_NOFILE) == 0) {
        status = avio_open(&output->pb, output_path, AVIO_FLAG_WRITE);
        if (status < 0) {
            cleanup();
            return fail(error, status, "open remux output");
        }
    }
    status = avformat_write_header(output, nullptr);
    if (status < 0) {
        cleanup();
        return fail(error, status, "write remux header");
    }
    header_written = true;
    video_packet = av_packet_alloc();
    audio_packet = av_packet_alloc();
    if (video_packet == nullptr || audio_packet == nullptr) {
        cleanup();
        return fail(error, AVERROR(ENOMEM), "allocate remux packets");
    }

    auto next_video = [&]() noexcept -> int {
        av_packet_unref(video_packet);
        for (;;) {
            const int result = av_read_frame(video, video_packet);
            if (result < 0 || video_packet->stream_index == video_index) {
                return result;
            }
            av_packet_unref(video_packet);
        }
    };
    auto next_audio = [&]() noexcept -> int {
        av_packet_unref(audio_packet);
        for (;;) {
            const int result = av_read_frame(source, audio_packet);
            if (result < 0 || (audio_packet->stream_index >= 0 &&
                               audio_mapping[audio_packet->stream_index] >= 0)) {
                return result;
            }
            av_packet_unref(audio_packet);
        }
    };
    int video_status = next_video();
    int audio_status = next_audio();
    while (video_status >= 0 || audio_status >= 0) {
        const bool write_video = audio_status < 0 ||
            (video_status >= 0 && av_compare_ts(
                video_packet->dts == AV_NOPTS_VALUE ? video_packet->pts : video_packet->dts,
                input_video->time_base,
                audio_packet->dts == AV_NOPTS_VALUE ? audio_packet->pts : audio_packet->dts,
                source->streams[audio_packet->stream_index]->time_base) <= 0);
        AVPacket *packet = write_video ? video_packet : audio_packet;
        const AVStream *input_stream = write_video
            ? input_video
            : source->streams[packet->stream_index];
        AVStream *output_stream = write_video
            ? output_video
            : output->streams[audio_mapping[packet->stream_index]];
        av_packet_rescale_ts(packet, input_stream->time_base, output_stream->time_base);
        packet->stream_index = output_stream->index;
        packet->pos = -1;
        status = av_interleaved_write_frame(output, packet);
        if (status < 0) {
            cleanup();
            return fail(error, status, "write remux packet");
        }
        if (write_video) {
            video_status = next_video();
        } else {
            audio_status = next_audio();
        }
    }
    if ((video_status != AVERROR_EOF && video_status < 0) ||
        (audio_status != AVERROR_EOF && audio_status < 0)) {
        status = video_status != AVERROR_EOF && video_status < 0 ? video_status : audio_status;
        cleanup();
        return fail(error, status, "read remux packet");
    }
    status = av_write_trailer(output);
    header_written = false;
    cleanup();
    return status < 0 ? fail(error, status, "write remux trailer") : 0;
}
