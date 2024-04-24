#include <iostream>

__global__ void matrixMulBasic(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    int size_A = M * N * sizeof(float);
    int size_B = N * K * sizeof(float);
    int size_C = M * K * sizeof(float);

    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C = (float *)malloc(size_C);

    for (int i = 0; i < M * N; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < N * K; ++i) h_B[i] = 2.0f;

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    matrixMulBasic<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
