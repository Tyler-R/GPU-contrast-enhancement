#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>
// helper for shared that are common to CUDA Samples
#include <helper_functions.h>
#include <helper_timer.h>

#include "hist-equ.cuh"

__global__ void convert_rgb_to_yuv( unsigned char *image_y_output, unsigned char *image_u_output, unsigned char *image_v_output,
                                    unsigned char *red_input, unsigned char *green_input, unsigned char *blue_input) {

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    image_y_output[index] = (unsigned char) (0.299 * red_input[index] + 0.587 * green_input[index] + 0.114 * blue_input[index]);
    image_u_output[index] = (unsigned char) (-0.169 * red_input[index] - 0.331 * green_input[index] + 0.499 * blue_input[index] + 128);
    image_v_output[index] = (unsigned char) (0.499 * red_input[index] - 0.418 * green_input[index] - 0.0813 * blue_input[index] + 128);
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG gpu_rgb2yuv(PPM_IMG color_image_input) {
    YUV_IMG yuv_image_output;

    yuv_image_output.w = color_image_input.w;
    yuv_image_output.h = color_image_input.h;

    int64_t image_size = yuv_image_output.w * yuv_image_output.h;

    yuv_image_output.img_y = (unsigned char *)malloc(sizeof(unsigned char) * image_size);
    yuv_image_output.img_u = (unsigned char *)malloc(sizeof(unsigned char) * image_size);
    yuv_image_output.img_v = (unsigned char *)malloc(sizeof(unsigned char) * image_size);

    unsigned char *gpu_y;
    unsigned char *gpu_u;
    unsigned char *gpu_v;

    unsigned char *gpu_red;
    unsigned char *gpu_green;
    unsigned char *gpu_blue;

    cudaMalloc(&gpu_red, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_green, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_blue, sizeof(unsigned char) * image_size);

    cudaMalloc(&gpu_y, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_u, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_v, sizeof(unsigned char) * image_size);

    cudaMemcpy(gpu_red, color_image_input.img_r, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_green, color_image_input.img_g, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_blue, color_image_input.img_b, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);

    int block_size = image_size / 256;
    int threads_per_block = 256;
    assert(image_size % 256 == 0);

    convert_rgb_to_yuv<<<block_size, threads_per_block>>>(gpu_y, gpu_u, gpu_v, gpu_red, gpu_green, gpu_blue);

    cudaMemcpy(yuv_image_output.img_y, gpu_y, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(yuv_image_output.img_u, gpu_u, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(yuv_image_output.img_v, gpu_v, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_red);
    cudaFree(gpu_green);
    cudaFree(gpu_blue);

    cudaFree(gpu_y);
    cudaFree(gpu_u);
    cudaFree(gpu_v);

    return yuv_image_output;
}

__device__ unsigned char clamp(int value, unsigned char min, unsigned char max) {
    return fmaxf(min, fminf(max, value));
}

__global__ void convert_yuv_to_rgb( unsigned char *red_output, unsigned char *green_output, unsigned char *blue_output,
                                    unsigned char *image_y_input, unsigned char *image_u_input, unsigned char *image_v_input) {

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    int y  = image_y_input[index];
    int cb = image_u_input[index] - 128;
    int cr = image_v_input[index] - 128;

    int temp_red   = (int) (y + 1.402 * cr);
    int temp_green = (int) (y - (0.344 * cb) - (0.714 * cr));
    int temp_blue  = (int) (y + (1.772 * cb));

    red_output[index]   = clamp(temp_red, 0, 255);
    green_output[index] = clamp(temp_green, 0, 255);
    blue_output[index]  = clamp(temp_blue, 0, 255);

}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG gpu_yuv2rgb(YUV_IMG yuv_image_input) {
    PPM_IMG color_image_output;


    color_image_output.w = yuv_image_input.w;
    color_image_output.h = yuv_image_input.h;

    int64_t image_size = color_image_output.w * color_image_output.h;

    color_image_output.img_r = (unsigned char *)malloc(sizeof(unsigned char) * image_size);
    color_image_output.img_g = (unsigned char *)malloc(sizeof(unsigned char) * image_size);
    color_image_output.img_b = (unsigned char *)malloc(sizeof(unsigned char) * image_size);

    unsigned char *gpu_red;
    unsigned char *gpu_green;
    unsigned char *gpu_blue;

    unsigned char *gpu_y;
    unsigned char *gpu_u;
    unsigned char *gpu_v;

    cudaMalloc(&gpu_red, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_green, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_blue, sizeof(unsigned char) * image_size);

    cudaMalloc(&gpu_y, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_u, sizeof(unsigned char) * image_size);
    cudaMalloc(&gpu_v, sizeof(unsigned char) * image_size);

    cudaMemcpy(gpu_y, yuv_image_input.img_y, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_u, yuv_image_input.img_u, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v, yuv_image_input.img_v, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);

    int block_size = image_size / 256;
    int threads_per_block = 256;
    assert(image_size % 256 == 0);

    convert_yuv_to_rgb<<<block_size, threads_per_block>>>(gpu_red, gpu_green, gpu_blue, gpu_y, gpu_u, gpu_v);

    cudaMemcpy(color_image_output.img_r, gpu_red, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(color_image_output.img_g, gpu_green, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(color_image_output.img_b, gpu_blue, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_red);
    cudaFree(gpu_green);
    cudaFree(gpu_blue);

    cudaFree(gpu_y);
    cudaFree(gpu_u);
    cudaFree(gpu_v);


    return color_image_output;
}
