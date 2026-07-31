#include <setjmp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

struct noperson_jpeg_error {
    struct jpeg_error_mgr base;
    jmp_buf jump;
};

static void noperson_jpeg_error_exit(j_common_ptr info) {
    struct noperson_jpeg_error* error = (struct noperson_jpeg_error*)info->err;
    longjmp(error->jump, 1);
}

int noperson_jpeg_roundtrip(
    const unsigned char* rgb,
    int width,
    int height,
    int quality,
    unsigned char* output
) {
    struct jpeg_compress_struct compressor = {0};
    struct noperson_jpeg_error compress_error = {0};
    unsigned char* encoded = NULL;
    unsigned long encoded_size = 0;
    compressor.err = jpeg_std_error(&compress_error.base);
    compress_error.base.error_exit = noperson_jpeg_error_exit;
    if (setjmp(compress_error.jump)) {
        jpeg_destroy_compress(&compressor);
        free(encoded);
        return 1;
    }
    jpeg_create_compress(&compressor);
    compressor.image_width = (JDIMENSION)width;
    compressor.image_height = (JDIMENSION)height;
    compressor.input_components = 3;
    compressor.in_color_space = JCS_RGB;
    jpeg_set_defaults(&compressor);
    jpeg_set_quality(&compressor, quality, TRUE);
    jpeg_mem_dest(&compressor, &encoded, &encoded_size);
    jpeg_start_compress(&compressor, TRUE);
    while (compressor.next_scanline < compressor.image_height) {
        JSAMPROW row = (JSAMPROW)(rgb + compressor.next_scanline * (JDIMENSION)(width * 3));
        jpeg_write_scanlines(&compressor, &row, 1);
    }
    jpeg_finish_compress(&compressor);
    jpeg_destroy_compress(&compressor);

    struct jpeg_decompress_struct decompressor = {0};
    struct noperson_jpeg_error decompress_error = {0};
    decompressor.err = jpeg_std_error(&decompress_error.base);
    decompress_error.base.error_exit = noperson_jpeg_error_exit;
    if (setjmp(decompress_error.jump)) {
        jpeg_destroy_decompress(&decompressor);
        free(encoded);
        return 2;
    }
    jpeg_create_decompress(&decompressor);
    jpeg_mem_src(&decompressor, encoded, encoded_size);
    jpeg_read_header(&decompressor, TRUE);
    decompressor.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decompressor);
    while (decompressor.output_scanline < decompressor.output_height) {
        JSAMPROW row = output + decompressor.output_scanline * (JDIMENSION)(width * 3);
        jpeg_read_scanlines(&decompressor, &row, 1);
    }
    jpeg_finish_decompress(&decompressor);
    jpeg_destroy_decompress(&decompressor);
    free(encoded);
    return 0;
}
