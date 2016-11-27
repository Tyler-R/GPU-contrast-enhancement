#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG gpu_contrast_enhancement_gray_image(PGM_IMG img_in)
{
    PGM_IMG result;
    int histogram_buffer[256];

    result.w = img_in.w;
    result.h = img_in.h;

    // image is a buffer where length = width * height
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    histogram(histogram_buffer, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img, img_in.img, histogram_buffer, result.w * result.h, 256);
    return result;
}
