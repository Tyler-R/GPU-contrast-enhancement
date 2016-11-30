#pragma once

#include "hist-equ.h"

PGM_IMG gpu_contrast_enhancement_gray_image(PGM_IMG img_in);
PPM_IMG gpu_contrast_enhancement_c_yuv(PPM_IMG img_in);
PPM_IMG gpu_contrast_enhancement_c_hsl(PPM_IMG img_in);

void gpu_make_histogram(int *histogram_output, unsigned char *img_in, int img_size, int nbr_bin);
void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin);

YUV_IMG gpu_rgb2yuv(PPM_IMG color_image_input);
PPM_IMG gpu_yuv2rgb(YUV_IMG yuv_image_input);

HSL_IMG gpu_rgb2hsl(PPM_IMG color_image_input);
PPM_IMG gpu_hsl2rgb(HSL_IMG img_in);
