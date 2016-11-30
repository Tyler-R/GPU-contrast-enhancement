#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.cuh"

PGM_IMG gpu_contrast_enhancement_gray_image(PGM_IMG img_in) {
    PGM_IMG result;
    int histogram_buffer[256];

    result.w = img_in.w;
    result.h = img_in.h;

    // image is a buffer where length = width * height
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    gpu_make_histogram(histogram_buffer, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img, img_in.img, histogram_buffer, result.w * result.h, 256);
    return result;
}

PPM_IMG gpu_contrast_enhancement_c_yuv(PPM_IMG img_in) {
    PPM_IMG result;

    int histogram_buffer[256];

    YUV_IMG yuv_temp_image = gpu_rgb2yuv(img_in);
    int64_t image_size = yuv_temp_image.h * yuv_temp_image.w;
    unsigned char *yuv_image_buffer = (unsigned char *)malloc(image_size * sizeof(unsigned char));

    gpu_make_histogram(histogram_buffer, yuv_temp_image.img_y, image_size, 256);
    gpu_histogram_equalization(yuv_image_buffer, yuv_temp_image.img_y, histogram_buffer, image_size, 256);

    free(yuv_temp_image.img_y);
    yuv_temp_image.img_y = yuv_image_buffer;

    result = gpu_yuv2rgb(yuv_temp_image);
    free(yuv_temp_image.img_y);
    free(yuv_temp_image.img_u);
    free(yuv_temp_image.img_v);

    return result;
}

PPM_IMG gpu_contrast_enhancement_c_hsl(PPM_IMG img_in) {
    PPM_IMG result;
    int histogram_buffer[256];

    HSL_IMG hsl_temp_image = gpu_rgb2hsl(img_in);
    int64_t image_size = hsl_temp_image.width * hsl_temp_image.height;

    unsigned char *hsl_image_buffer = (unsigned char *)malloc(image_size * sizeof(unsigned char));

    gpu_make_histogram(histogram_buffer, hsl_temp_image.l, image_size, 256);
    gpu_histogram_equalization(hsl_image_buffer, hsl_temp_image.l, histogram_buffer, image_size, 256);

    free(hsl_temp_image.l);
    hsl_temp_image.l = hsl_image_buffer;

    result = gpu_hsl2rgb(hsl_temp_image);
    free(hsl_temp_image.h);
    free(hsl_temp_image.s);
    free(hsl_temp_image.l);
    return result;
}
