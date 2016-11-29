#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>

#include "hist-equ.cuh"

__global__ convert_rgb_to_yuv() {

}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG gpu_rgb2yuv(PPM_IMG image_input) {
    YUV_IMG image_output;
    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;

    image_output.w = image_input.w;
    image_output.h = image_input.h;

    int64_t image_size = image_output.w * image_output.h;

    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char) * image_size);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char) * image_size);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char) * image_size);


    for(i = 0; i < image_size; i ++){
        r = image_input.img_r[i];
        g = image_input.img_g[i];
        b = image_input.img_b[i];

        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);

        img_out.img_y[i] = y;
        img_out.img_u[i] = cb;
        img_out.img_v[i] = cr;
    }

    return img_out;
}

__global__ void convert_yuv_to_rgb(unsigned char *red, unsigned char *green, unsigned char *blue,
                                   unsigned char *img_y, unsigned char *img_u, unsigned char *img_v) {

    int y   = img_y[index];
    int cb  = img_u[index] - 128;
    int cr  = img_v[index] - 128;

    int temp_red    = (int)(y + 1.402 * cr);
    int temp_green  = (int)(y - 0.344 * cb - 0.714 * cr);
    int temp_blue   = (int)(y + 1.772 * cb);

    red[i] = clamp(temp_red, 0, 255);
    green[i] = clamp(temp_green, 0, 255);
    blue[i] = clamp(temp_blue, 0, 255);

}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG gpu_yuv2rgb(YUV_IMG yuv_image_input) {
    PPM_IMG color_image_output;
    int i;
    int rt,gt,bt;
    int y, cb, cr;


    color_image_output.w = yuv_image_input.w;
    color_image_output.h = yuv_image_input.h;

    int64_t image_size = color_image_output.w * color_image_output.h

    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char) * image_size);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char) * image_size);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char) * image_size);

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

    convert_yuv_to_rgb<<<1,1>>>(gpu_red, gpu_green, gpu_blue, gpu_y, gpu_u, gpu_v);

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
