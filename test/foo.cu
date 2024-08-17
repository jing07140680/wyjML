#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel function to run on the GPU
__global__ void fooKernel(float* input, float* output, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        output[index] = input[index] * 2.0f;  // Example computation
    }
}

// Function to handle the GPU execution
void foo(float* input, float* output, int size) {
    float *d_input, *d_output;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    cudaEventRecord(start, 0);
    // Copy data from host to device
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    // Launch the kernel on the GPU
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (size + blockSize - 1) / blockSize;  // Number of blocks
    fooKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    // Copy the result back to the host
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int size = 1024;
    float* input = new float[size];
    float* output = new float[size];

    // Initialize input data
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(i);
    }

    // Call the attention function that uses the GPU
    foo(input, output, size);

    // Output the first 10 results to verify correctness
    for (int i = 0; i < 10; ++i) {
        std::cout << output[i] << std::endl;
    }

    // Clean up
    delete[] input;
    delete[] output;

    return 0;
}
