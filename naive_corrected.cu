#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define N (1 << 24)
#define THREADS 256

// CPU reduction
float cpuReduction(float *a, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += a[i];
    }

    return sum;
}

// Naive CUDA reduction using interleaved addressing
__global__ void naiveReduction(float *input, float *output, int n) {

    __shared__ float sdata[THREADS];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? input[i] : 0.0f;

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {

        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Repeat reduction stages until one sum remains
float* launchNaive(float *d_input,
                   float *d_temp1,
                   float *d_temp2,
                   int n) {

    float *current = d_input;
    float *next = d_temp1;

    while (n > 1) {

        int blocks = (n + THREADS - 1) / THREADS;

        naiveReduction<<<blocks, THREADS>>>(current, next, n);

        n = blocks;
        current = next;

        next = (next == d_temp1) ? d_temp2 : d_temp1;
    }

    return current;
}

int main() {

    float *h_input = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // CPU calculation and timing
    auto cpuStart = std::chrono::high_resolution_clock::now();

    float cpuSum = cpuReduction(h_input, N);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart).count();

    // Allocate GPU memory
    float *d_input, *d_temp1, *d_temp2;

    int maxBlocks = (N + THREADS - 1) / THREADS;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_temp1, maxBlocks * sizeof(float));
    cudaMalloc(&d_temp2, maxBlocks * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float),
               cudaMemcpyHostToDevice);

    // Warm-up
    float *d_result = launchNaive(
        d_input, d_temp1, d_temp2, N);

    cudaDeviceSynchronize();

    // CUDA event timing
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int repetitions = 10;

    cudaEventRecord(start);

    for (int r = 0; r < repetitions; r++) {
        d_result = launchNaive(
            d_input, d_temp1, d_temp2, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalTime = 0.0f;

    cudaEventElapsedTime(&totalTime, start, stop);

    float gpuTime = totalTime / repetitions;

    // Copy final GPU result to CPU
    float gpuSum;

    cudaMemcpy(&gpuSum, d_result, sizeof(float),
               cudaMemcpyDeviceToHost);

    // Display results
    printf("\n----- CORRECTED NAIVE RESULTS -----\n");

    printf("CPU Sum          : %.0f\n", cpuSum);
    printf("Naive GPU Sum    : %.0f\n", gpuSum);

    printf("CPU Time         : %.4f ms\n", cpuTime);
    printf("Naive GPU Time   : %.4f ms\n", gpuTime);

    printf("Speedup          : %.2f times\n",
           cpuTime / gpuTime);

    printf("Results Match    : %s\n",
           (cpuSum == gpuSum) ? "YES" : "NO");

    // Free memory
    cudaFree(d_input);
    cudaFree(d_temp1);
    cudaFree(d_temp2);

    free(h_input);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
