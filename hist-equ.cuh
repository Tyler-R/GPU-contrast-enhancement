#pragma once

#include "hist-equ.h"

PGM_IMG gpu_contrast_enhancement_gray_image(PGM_IMG img_in);

void gpu_make_histogram(int *histogram_output, unsigned char *img_in, int img_size, int nbr_bin);
void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin);
