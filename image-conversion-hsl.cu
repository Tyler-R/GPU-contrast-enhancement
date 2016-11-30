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
#include "gpu-memory.cuh"


__global__ void convert_rgb_to_hsl( float *image_h_output, float *image_s_output, unsigned char *image_l_output,
                                    unsigned char *image_red_input, unsigned char *image_green_input, unsigned char *image_blue_input) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    float h, s, l;

    //Convert RGB to [0,1]
    float red   = ((float) image_red_input[index] / 255);
    float green = ((float) image_green_input[index] / 255);
    float blue  = ((float) image_blue_input[index] / 255);

    float min_color_value = fminf(red, fminf(green, blue));
    float max_color_value = fmaxf(red, fmaxf(green, blue));

    float color_value_delta = max_color_value - min_color_value;

    l = (max_color_value + min_color_value) / 2;


    if ( color_value_delta == 0 )//This is a gray, no chroma...
    {
        h = 0;
        s = 0;
    }
    else                                    //Chromatic data...
    {
        if ( l < 0.5 ) {
            s = color_value_delta / (max_color_value + min_color_value);
        } else {
            s = color_value_delta / (2 - max_color_value - min_color_value);
        }

        float red_delta = (((max_color_value - red) / 6) + (color_value_delta / 2)) / color_value_delta;
        float green_delta = (((max_color_value - green) / 6) + (color_value_delta / 2)) / color_value_delta;
        float blue_delta = (((max_color_value - blue) / 6) + (color_value_delta / 2)) / color_value_delta;

        if( red == max_color_value ){
            h = blue_delta - green_delta;
        } else {
            if( green == max_color_value ){
                h = (1.0 / 3.0) + red_delta - blue_delta;
            } else { // blue is max color
                h = (2.0/3.0) + green_delta - red_delta;
            }
        }
    }

    if (h < 0) {
        h += 1;
    }
    if (h > 1) {
        h -= 1;
    }

    image_h_output[index] = h;
    image_s_output[index] = s;
    image_l_output[index] = (unsigned char)(l * 255);



}

//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG gpu_rgb2hsl(PPM_IMG color_image_input) {

    HSL_IMG hsl_image_output;

    hsl_image_output.width  = color_image_input.w;
    hsl_image_output.height = color_image_input.h;

    int64_t image_size = hsl_image_output.width * hsl_image_output.height;

    hsl_image_output.h = (float *) malloc(image_size * sizeof(float));
    hsl_image_output.s = (float *) malloc(image_size * sizeof(float));
    hsl_image_output.l = (unsigned char *) malloc(image_size * sizeof(unsigned char));

    // use custom GpuMemory class to simplify the creation, deletion and copying of GPU buffers.
    GpuMemory<float *, float> gpu_h(image_size);
    GpuMemory<float *, float> gpu_s(image_size);
    GpuMemory<unsigned char *, unsigned char> gpu_l(image_size);

    GpuMemory<unsigned char *, unsigned char> gpu_red(image_size);
    GpuMemory<unsigned char *, unsigned char> gpu_green(image_size);
    GpuMemory<unsigned char *, unsigned char> gpu_blue(image_size);

    gpu_red.copyToGpuData(color_image_input.img_r, image_size);
    gpu_green.copyToGpuData(color_image_input.img_g, image_size);
    gpu_blue.copyToGpuData(color_image_input.img_b, image_size);

    int block_size = image_size / 256;
    int threads_per_block = 256;
    convert_rgb_to_hsl<<<block_size, threads_per_block>>>(
            gpu_h.getArray(), gpu_s.getArray(), gpu_l.getArray(),
            gpu_red.getArray(), gpu_green.getArray(), gpu_blue.getArray()
    );

    gpu_h.getDataFromGpu(hsl_image_output.h, image_size);
    gpu_s.getDataFromGpu(hsl_image_output.s, image_size);
    gpu_l.getDataFromGpu(hsl_image_output.l, image_size);

    return hsl_image_output;
}

float gpu_Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
PPM_IMG gpu_hsl2rgb(HSL_IMG img_in) {
    int i;
    PPM_IMG result;

    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    for(i = 0; i < img_in.width*img_in.height; i ++){
        float H = img_in.h[i];
        float S = img_in.s[i];
        float L = img_in.l[i]/255.0f;
        float var_1, var_2;

        unsigned char r,g,b;

        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {

            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            r = 255 * gpu_Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * gpu_Hue_2_RGB( var_1, var_2, H );
            b = 255 * gpu_Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
        }
        result.img_r[i] = r;
        result.img_g[i] = g;
        result.img_b[i] = b;
    }

    return result;
}
