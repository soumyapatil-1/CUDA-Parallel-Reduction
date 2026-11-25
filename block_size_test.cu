#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 24)

__global__ void optimizedReduction(
    float *input, float *output, int n) {

    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;

    unsigned int i =
        blockIdx.x * (blockSize * 2) + tid;

    float sum = 0.0f;

    if (i < n)
        sum += input[i];

    if (i + blockSize < n)
        sum += input[i + blockSize];

    sdata[tid] = sum;
    __syncthreads();

    // Sequential addressing
    for (unsigned int s = blockSize / 2; s >= 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    // Warp shuffle
    if (tid < 32) {
        float value = sdata[tid];

        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(
                0xffffffff, value, offset);
        }

        if (tid == 0)
            output[blockIdx.x] = value;
    }
}

// Complete reduction on GPU
float* runReduction(
    float *d_input,
    float *d_temp1,
    float *d_temp2,
    int n,
    int threads) {

    float *current = d_input;
    float *next = d_temp1;

    while (n > 1) {

        int blocks = (n + threads * 2 - 1)
                     / (threads * 2);

        size_t sharedMemory = threads * sizeof(float);

        optimizedReduction<<<blocks, threads, sharedMemory>>>(
            current, next, n);

        n = blocks;
        current = next;

        next = (next == d_temp1) ? d_temp2 : d_temp1;
    }

    return current;
}

int main() {

    float *h_input =
        (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    float *d_input, *d_temp1, *d_temp2;

    cudaMalloc(&d_input, N * sizeof(float));

    cudaMemcpy(d_input, h_input,
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    int threadSizes[] = {128, 256, 512};
    int repetitions = 10;

    printf("\n===== BLOCK SIZE COMPARISON =====\n");
    printf("Input size: %d elements\n", N);
    printf("GPU: NVIDIA T4 (Colab)\n\n");

    printf("Threads/Block     GPU Time (ms)     GPU Sum\n");

    for (int k = 0; k < 3; k++) {

        int threads = threadSizes[k];

        int maxBlocks =
            (N + threads * 2 - 1) / (threads * 2);

        cudaMalloc(&d_temp1,
                   maxBlocks * sizeof(float));

        cudaMalloc(&d_temp2,
                   maxBlocks * sizeof(float));

        // Warm-up
        float *d_result = runReduction(
            d_input, d_temp1, d_temp2, N, threads);

        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        for (int r = 0; r < repetitions; r++) {
            d_result = runReduction(
                d_input, d_temp1, d_temp2, N, threads);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float totalTime = 0.0f;

        cudaEventElapsedTime(
            &totalTime, start, stop);

        float averageTime = totalTime / repetitions;

        float gpuSum;

        cudaMemcpy(&gpuSum, d_result,
                   sizeof(float),
                   cudaMemcpyDeviceToHost);

        printf("%-18d %-18.4f %.0f\n",
               threads, averageTime, gpuSum);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_temp1);
        cudaFree(d_temp2);
    }

    cudaFree(d_input);
    free(h_input);

    return 0;
}
