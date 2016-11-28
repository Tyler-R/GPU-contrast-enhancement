#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>

#include "hist-equ.cuh"

__global__ void zero_out_array(int *output_array) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x - 1;
  output_array[index] = 0;
}

void gpu_make_histogram(int *histogram_output, unsigned char *img_in, int img_size, int nbr_bin){


    int number_of_blocks = (int) (nbr_bin / 256);

    // make sure number of blocks is a multiple of 256
    assert((nbr_bin % 256) == 0);
    assert(number_of_blocks > 0);

    zero_out_array<<<number_of_blocks, nbr_bin>>>(histogram_output);

    for (int i = 0; i < img_size; i ++){
        histogram_output[img_in[i]] ++;
    }

    //cuda_free();
}

void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }


    }

    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }

    }
}
