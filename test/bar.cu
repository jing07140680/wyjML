#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 16
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel for matrix multiplication (C = A * B)
__global__ void matMulKernel(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float value = 0.0f;
        for (int k = 0; k < N; ++k)
        {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// Function to perform matrix multiplication on the GPU
void matMulGPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N)
{
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, N * N * sizeof(float)));

    // Copy matrices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the matrix multiplication kernel
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}

int main()
{
    int N = 1024; // Size of the NxN matrix
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);

    // Perform matrix multiplication on the GPU
    matMulGPU(A, B, C, N);

    // Verify the result
    bool correct = true;
    for (int i = 0; i < N * N; ++i)
    {
        if (C[i] != N * 2.0f)
        {
            correct = false;
            break;
        }
    }

    if (correct)
    {
        std::cout << "Matrix multiplication is correct!" << std::endl;
    }
    else
    {
        std::cout << "Matrix multiplication is incorrect!" << std::endl;
    }

    return 0;
}
