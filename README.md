# Parallel Reduction in CUDA – Local GPU Implementation

## 1. Overview

This project demonstrates Parallel Reduction using CUDA to calculate the sum of a large array of floating-point numbers using an NVIDIA GPU.

The main aim is to understand how parallel processing improves computation speed by distributing work among multiple GPU threads.

## 2. Objectives

- Implement a CPU-based sum for reference.
- Implement a Naive CUDA Reduction.
- Implement an Optimised CUDA Reduction.
- Understand shared memory and thread synchronisation.
- Compare different reduction approaches.
- Verify the correctness of the calculated sum.

## 3. Technologies Used

- CUDA C++
- NVIDIA CUDA Toolkit
- NVIDIA GPU
- `nvcc` CUDA compiler
- Windows Batch / Linux Shell scripts

## 4. Project Structure

```text
Local-GPU-Implementation/
│
├── src/
│   └── parallel_reduction.cu
│
├── scripts/
│   ├── build_linux.sh
│   └── build_windows.bat
│
└── results/
    └── results_template.csv
```

## 5. Implementation

### A. CPU Reduction
The CPU calculates the sum of all array elements sequentially using a loop. This result is used as a reference for checking GPU results.

### B. Naive GPU Reduction
- Uses shared memory to store values.
- Uses interleaved addressing.
- Uses the condition `tid % (2*s) == 0`.
- Uses `__syncthreads()` to synchronise threads.

### C. Optimised GPU Reduction
- Uses sequential addressing.
- Adds two elements during the loading stage.
- Uses warp shuffle with `__shfl_down_sync()`.
- Reduces unnecessary synchronisation and improves thread utilisation.

## 6. Building and Running

### Windows

Open the x64 Native Tools Command Prompt for Visual Studio.

Navigate to the project folder and execute:

```bat
scripts\build_windows.bat
```

To build and run:

```bat
scripts\build_windows.bat run
```

### Linux

Build the project:

```bash
bash scripts/build_linux.sh
```

Build and run:

```bash
bash scripts/build_linux.sh --run
```

**Requirement:** An NVIDIA GPU with a compatible NVIDIA driver and CUDA Toolkit is required.

## 7. Result Verification

The program compares the GPU reduction result with the CPU reference result.

It also performs correctness checks for different input sizes and provides a benchmark results CSV when executed successfully.

Actual GPU execution times, speedup and block-size results should be added after running the experiment on the target NVIDIA GPU.

## 8. Conclusion

This implementation demonstrates how CUDA parallel reduction can be used to perform large-scale summation on an NVIDIA GPU. It compares naive and optimised approaches and explains how memory access, thread organisation and warp-level operations affect performance.

## 9. Source Acknowledgement

This implementation is based on the CUDA evaluation project shared by Bhoomi Bankapur, Sneha Shettar and Sanket Ullagaddi. Credit is retained for the original work.
