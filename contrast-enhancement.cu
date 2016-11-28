#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.cuh"

PGM_IMG gpu_contrast_enhancement_gray_image(PGM_IMG img_in)
{
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

PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    PPM_IMG result;

    int histogram_buffer[256];

    YUV_IMG yuv_temp_image = rgb2yuv(img_in);
    unsigned char *yuv_image_buffer = (unsigned char *)malloc(yuv_temp_image.h * yuv_temp_image.w * sizeof(unsigned char));

    histogram(histogram_buffer, yuv_temp_image.img_y, yuv_temp_image.h * yuv_temp_image.w, 256);
    histogram_equalization(yuv_image_buffer, yuv_temp_image.img_y, hist,yuv_temp_image.h * yuv_temp_image.w, 256);

    free(yuv_temp_image.img_y);
    yuv_temp_image.img_y = yuv_image_buffer;

    result = yuv2rgb(yuv_temp_image);
    free(yuv_temp_image.img_y);
    free(yuv_temp_image.img_u);
    free(yuv_temp_image.img_v);

    return result;
}
