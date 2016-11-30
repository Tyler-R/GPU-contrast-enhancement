#pragma once
// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>
// helper for shared that are common to CUDA Samples
#include <helper_functions.h>
#include <helper_timer.h>


template <typename array_pointer_type, typename array_data_type>
class GpuMemory {
    public:
        GpuMemory(int size) {
            cudaMalloc(&array, sizeof(array_data_type) * size);
            array_size = size;
        }

        ~GpuMemory() {
            cudaFree(array);
        }

        array_pointer_type getArray() {
            return array;
        }

        void copyToGpuData(array_pointer_type source, int size) {
            cudaMemcpy(array, source, sizeof(array_data_type) * size, cudaMemcpyHostToDevice);
        }

        void getDataFromGpu(array_pointer_type destination, int size) {
            cudaMemcpy(destination, array, sizeof(array_data_type) * size, cudaMemcpyDeviceToHost);
        }
    private:
        array_pointer_type array;
        int array_size;

};
