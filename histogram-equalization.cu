#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>

#include "hist-equ.cuh"

__global__ void zero_out_buffer(int *output_buffer) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  output_buffer[index] = 0;
}

__global__ void make_histogram(int *histogram_output_buffer, unsigned char *image_buffer) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int image_value_at_index = image_buffer[index];
  // increment index by 1
  atomicAdd(&histogram_output_buffer[image_value_at_index], 1);
}

void gpu_make_histogram(int *histogram_output, unsigned char *image_input, int image_size, int histogram_size){

    int *gpu_histogram_output;
    unsigned char *gpu_image_input;

    cudaMalloc(&gpu_histogram_output, sizeof(int) * histogram_size);
    cudaMalloc(&gpu_image_input, sizeof(unsigned char) * image_size);

    cudaMemcpy(gpu_histogram_output, histogram_output, sizeof(int) * histogram_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_image_input, image_input, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);


    int number_of_blocks = (int) (histogram_size / 256);

    // make sure number of blocks is a multiple of 256
    assert((histogram_size % 256) == 0);
    assert(number_of_blocks > 0);

    zero_out_buffer<<<number_of_blocks, histogram_size>>>(gpu_histogram_output);
    make_histogram<<<number_of_blocks, histogram_size>>>(gpu_histogram_output, gpu_image_input);

    cudaMemcpy(histogram_output, gpu_histogram_output, sizeof(int) * histogram_size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_histogram_output);
    cudaFree(gpu_image_input);

}

__global__ void get_result_image(unsigned char *image_output, unsigned char *image_input, int *lookup_table) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    int lookup_value = lookup_table[image_input[index]];

    image_output[index] = lookup_value > 255 ? 255 : (unsigned char)lookup_value;

}

void gpu_histogram_equalization(unsigned char *image_output, unsigned char *image_input,
                            int * histogram_input, int image_size, int histogram_size) {

    // NOTE: "LUT" is a common abbreviation for "lookup table"
    int *lookup_table = (int *)malloc(sizeof(int) * histogram_size);

    /* Construct the LUT by calculating the CDF */
    int cdf = 0; // cumulative distribution function???
    int min = 0;
    int i = 0;
    while(min == 0){
        min = histogram_input[i++];
    }
    int d = image_size - min;


    for(i = 0; i < histogram_size; i ++) {
        cdf += histogram_input[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lookup_table[i] = (int)(((float)cdf - min) * 255 / d + 0.5);
        if(lookup_table[i] < 0){
            lookup_table[i] = 0;
        }
    }

    unsigned char *gpu_image_output;
    unsigned char *gpu_image_input;
    int *gpu_lookup_table;

    cudaMalloc(&gpu_image_output, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_image_input, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_lookup_table, sizeof(int) * histogram_size);

    cudaMemcpy(gpu_image_output, image_output, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_image_input, image_input, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_lookup_table, lookup_table, sizeof(int) * histogram_size, cudaMemcpyHostToDevice);

    int number_of_blocks = image_size / histogram_size;
    int number_of_threads = histogram_size;

    assert(image_size % histogram_size == 0);

    get_result_image<<<number_of_blocks, number_of_threads>>>(gpu_image_output, gpu_image_input, gpu_lookup_table);

    cudaMemcpy(image_output, gpu_image_output, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_image_output);
    cudaFree(gpu_image_input);
    cudaFree(gpu_lookup_table);

}