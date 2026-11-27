// =============================================================================
//  Parallel Reduction in CUDA  —  CUDA Lab Evaluation, Theme 1
// 
//  Compares three ways of summing 2^24 floats:
//    1. CPU sequential sum (reference, double accumulator)
//    2. Naive GPU kernel     : interleaved addressing, tid % (2*s) == 0
//    3. Optimised GPU kernel : sequential addressing + first add during load
//                              + warp shuffle (__shfl_down_sync)
//  Every GPU method reduces partial sums ON THE GPU (multi-pass) until one
//  value remains. Kernels are timed with CUDA events (warm-up + averaged runs).
//
//  Build : nvcc -O3 -std=c++17 -arch=native src/parallel_reduction.cu -o parallel_reduction
//  Run   : ./parallel_reduction            (see README.md for options)
// =============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Error checking
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d\n  %s\n  -> %s (%s)\n",   \
                         __FILE__, __LINE__, #call, cudaGetErrorString(err__),  \
                         cudaGetErrorName(err__));                              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

typedef unsigned long long u64;

// Relative tolerance used to compare a float GPU sum with the double CPU sum.
// Float has ~7 significant digits; a tree sum of 2^24 values typically differs
// from the exact sum by ~1e-7..1e-6 relative, so 1e-5 is a safe, still strict, limit.
static const double kRelTol = 1e-5;

// ============================================================================
//  KERNEL 1 — NAIVE: interleaved addressing with divergent branches
//  Each block loads blockDim.x elements into shared memory, then at step s
//  only threads with tid % (2*s) == 0 add their neighbour s positions away.
// ============================================================================
__global__ void reduceNaive(const float* __restrict__ in, float* __restrict__ out, u64 n)
{
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const u64 i = (u64)blockIdx.x * blockDim.x + tid;      // 64-bit index: no overflow

    sdata[tid] = (i < n) ? in[i] : 0.0f;                   // pad tail with 0 (identity of +)
    __syncthreads();                                       // all loads visible to the block

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {                          // divergent: active lanes are scattered
            sdata[tid] += sdata[tid + s];                  // interleaved (strided) addressing
        }
        __syncthreads();                                   // finish step s before step 2s
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];              // one partial sum per block
}

// ============================================================================
//  KERNEL 2 — OPTIMISED
//   (a) first add during load : each thread loads TWO elements and adds them,
//       so a block covers 2*blockDim.x elements and no thread is idle at start
//   (b) sequential addressing : active threads are 0..s-1 (contiguous), so whole
//       warps stay active/idle together (no divergence) and accesses have
//       stride 1 (no shared-memory bank conflicts)
//   (c) warp shuffle          : the last 64 -> 1 values are reduced inside one
//       warp with __shfl_down_sync, register to register, with no shared
//       memory and no __syncthreads()
//  Requirement: blockDim.x is a power of two and >= 64 (checked on the host).
// ============================================================================
__global__ void reduceOptimised(const float* __restrict__ in, float* __restrict__ out, u64 n)
{
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const u64 i = (u64)blockIdx.x * (2ull * blockDim.x) + tid;

    // (a) first add during load (bounds-checked for sizes that are not multiples)
    float sum = 0.0f;
    if (i < n)              sum  = in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // (b) sequential addressing in shared memory, down to 64 values
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sum += sdata[tid + s];
            sdata[tid] = sum;
        }
        __syncthreads();
    }

    // (c) final warp: fold 64 -> 32 from shared memory, then shuffle 32 -> 1
    if (tid < 32) {
        sum += sdata[tid + 32];                            // valid because blockDim.x >= 64
        const unsigned int fullMask = 0xffffffffu;         // all 32 lanes participate
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(fullMask, sum, offset);
        }
        if (tid == 0) out[blockIdx.x] = sum;
    }
}

// ----------------------------------------------------------------------------
// Host-side multi-pass driver
// ----------------------------------------------------------------------------
enum class Method { Naive, Optimised };

static const char* methodName(Method m) { return m == Method::Naive ? "GPU naive" : "GPU optimised"; }

static u64 elemsPerBlock(Method m, unsigned int bs) { return m == Method::Naive ? (u64)bs : 2ull * bs; }

static u64 ceilDiv(u64 a, u64 b) { return (a + b - 1) / b; }

// Runs all passes on the GPU. d_in is never modified. Result ends in *d_result
// (a pointer into bufA or bufB). Returns number of passes.
static int launchReduction(Method m, const float* d_in, u64 n, unsigned int bs,
                           float* bufA, float* bufB, const float** d_result)
{
    const size_t smem = (size_t)bs * sizeof(float);
    const float* src = d_in;
    float* dst = bufA;
    int passes = 0;

    do {
        const u64 blocks = ceilDiv(n, elemsPerBlock(m, bs));
        if (m == Method::Naive) reduceNaive<<<(unsigned int)blocks, bs, smem>>>(src, dst, n);
        else                    reduceOptimised<<<(unsigned int)blocks, bs, smem>>>(src, dst, n);
        CUDA_CHECK_LAUNCH();
        ++passes;
        src = dst;
        dst = (dst == bufA) ? bufB : bufA;
        n = blocks;
    } while (n > 1);

    *d_result = src;
    return passes;
}

struct GpuResult {
    Method method;
    unsigned int block;
    double avgMs, minMs, maxMs;
    double sum;
    int passes;
    bool ok;
    double relErr;
};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
static void fillDeterministic(std::vector<float>& h, unsigned int seed)
{
    // std::mt19937 is bit-exact on every compiler; we map 24 random bits to
    // [0,1) ourselves so Windows and Linux builds generate identical data.
    std::mt19937 rng(seed);
    for (size_t k = 0; k < h.size(); ++k)
        h[k] = (float)(rng() >> 8) * (1.0f / 16777216.0f);
}

static double cpuSum(const float* a, size_t n)
{
    double s = 0.0;                                        // double accumulator = accurate reference
    for (size_t k = 0; k < n; ++k) s += a[k];
    return s;
}

static double relError(double got, double ref)
{
    const double denom = std::fabs(ref) > 1e-30 ? std::fabs(ref) : 1.0;
    return std::fabs(got - ref) / denom;
}

static bool isPow2(unsigned int x) { return x && !(x & (x - 1)); }

static double gbPerSec(u64 n, double ms) { return ((double)n * sizeof(float)) / (ms * 1e-3) / 1e9; }

struct Options {
    u64 n = 1ull << 24;
    unsigned int block = 256;
    int iters = 30;
    int warmup = 5;
    bool sweep = true;
    bool selftest = true;
    unsigned int seed = 2024;
    std::string csv = "results/benchmark_results.csv";
};

static void usage(const char* p)
{
    std::printf("Usage: %s [--n N] [--block B] [--iters K] [--warmup W] [--seed S]\n"
                "          [--csv FILE] [--no-sweep] [--no-selftest]\n"
                "  --n       number of floats (default 16777216 = 2^24)\n"
                "  --block   threads per block for the main table (power of 2, >= 64; default 256)\n"
                "  --iters   timed iterations (default 30)    --warmup  warm-up runs (default 5)\n"
                "  --csv     output CSV (default results/benchmark_results.csv)\n", p);
}

static Options parseArgs(int argc, char** argv)
{
    Options o;
    for (int a = 1; a < argc; ++a) {
        auto need = [&](const char* f) -> const char* {
            if (a + 1 >= argc) { std::fprintf(stderr, "Missing value for %s\n", f); std::exit(EXIT_FAILURE); }
            return argv[++a];
        };
        if      (!std::strcmp(argv[a], "--n"))           o.n = std::strtoull(need("--n"), nullptr, 10);
        else if (!std::strcmp(argv[a], "--block"))       o.block = (unsigned)std::strtoul(need("--block"), nullptr, 10);
        else if (!std::strcmp(argv[a], "--iters"))       o.iters = std::atoi(need("--iters"));
        else if (!std::strcmp(argv[a], "--warmup"))      o.warmup = std::atoi(need("--warmup"));
        else if (!std::strcmp(argv[a], "--seed"))        o.seed = (unsigned)std::strtoul(need("--seed"), nullptr, 10);
        else if (!std::strcmp(argv[a], "--csv"))         o.csv = need("--csv");
        else if (!std::strcmp(argv[a], "--no-sweep"))    o.sweep = false;
        else if (!std::strcmp(argv[a], "--no-selftest")) o.selftest = false;
        else if (!std::strcmp(argv[a], "--help") || !std::strcmp(argv[a], "-h")) { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Unknown option: %s\n", argv[a]); usage(argv[0]); std::exit(EXIT_FAILURE); }
    }
    if (o.n < 1)      { std::fprintf(stderr, "--n must be >= 1\n"); std::exit(EXIT_FAILURE); }
    if (o.iters < 1)  o.iters = 1;
    if (o.warmup < 0) o.warmup = 0;
    return o;
}

// Checks a block size against the device limits; prints the reason if invalid.
static bool blockSizeValid(unsigned int bs, const cudaDeviceProp& p, bool verbose)
{
    const char* why = nullptr;
    if (!isPow2(bs) || bs < 64)                           why = "must be a power of two >= 64";
    else if ((int)bs > p.maxThreadsPerBlock)              why = "exceeds maxThreadsPerBlock";
    else if ((int)bs > p.maxThreadsDim[0])                why = "exceeds maxThreadsDim.x";
    else if ((size_t)bs * sizeof(float) > p.sharedMemPerBlock) why = "exceeds sharedMemPerBlock";
    if (why && verbose) std::printf("  [skip] block size %u: %s\n", bs, why);
    return why == nullptr;
}

// Times one method at one block size with CUDA events (all passes included).
static GpuResult benchmark(Method m, const float* d_in, u64 n, unsigned int bs,
                           float* bufA, float* bufB, double ref, const Options& o,
                           cudaEvent_t evStart, cudaEvent_t evStop)
{
    GpuResult r{};
    r.method = m; r.block = bs;
    const float* d_res = nullptr;

    for (int w = 0; w < o.warmup; ++w) launchReduction(m, d_in, n, bs, bufA, bufB, &d_res);
    CUDA_CHECK(cudaDeviceSynchronize());

    double total = 0.0;
    r.minMs = 1e30; r.maxMs = 0.0;
    for (int it = 0; it < o.iters; ++it) {
        CUDA_CHECK(cudaEventRecord(evStart));
        r.passes = launchReduction(m, d_in, n, bs, bufA, bufB, &d_res);
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        total += ms;
        r.minMs = std::min(r.minMs, (double)ms);
        r.maxMs = std::max(r.maxMs, (double)ms);
    }
    r.avgMs = total / o.iters;

    float h = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h, d_res, sizeof(float), cudaMemcpyDeviceToHost));
    r.sum = h;
    r.relErr = relError(r.sum, ref);
    r.ok = r.relErr <= kRelTol;
    return r;
}

// Correctness on awkward sizes (not multiples of the block width).
static bool selfTest(const std::vector<unsigned int>& blocks, unsigned int seed)
{
    const u64 sizes[] = {1, 2, 31, 63, 64, 65, 255, 1000, 4097, 65537, 1000003};
    std::vector<float> h(1000003);
    fillDeterministic(h, seed + 7);

    float *d_in = nullptr, *a = nullptr, *b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a, h.size() * sizeof(float)));  // big enough for any pass
    CUDA_CHECK(cudaMalloc(&b, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

    bool allOk = true;
    int tests = 0;
    for (u64 n : sizes) {
        const double ref = cpuSum(h.data(), (size_t)n);
        for (unsigned int bs : blocks) {
            for (Method m : {Method::Naive, Method::Optimised}) {
                const float* d_res = nullptr;
                launchReduction(m, d_in, n, bs, a, b, &d_res);
                float g = 0.0f;
                CUDA_CHECK(cudaMemcpy(&g, d_res, sizeof(float), cudaMemcpyDeviceToHost));
                const double e = relError(g, ref);
                ++tests;
                if (e > kRelTol) {
                    allOk = false;
                    std::printf("  FAIL %-13s n=%-8llu block=%-4u gpu=%.6f cpu=%.6f relErr=%.2e\n",
                                methodName(m), n, bs, g, ref, e);
                }
            }
        }
    }
    std::printf("  %d edge-size checks (n = 1 ... 1000003, every block size, both kernels): %s\n",
                tests, allOk ? "ALL PASS" : "SOME FAILED");
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    return allOk;
}

static std::string csvSafe(std::string s)
{
    for (char& c : s) if (c == ',' || c == '"') c = ' ';
    return s;
}

// ============================================================================
int main(int argc, char** argv)
{
    const Options opt = parseArgs(argc, argv);

    // ---------------- Device information ----------------
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) { std::fprintf(stderr, "No CUDA-capable GPU found.\n"); return EXIT_FAILURE; }
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int memClockKHz = 0, busWidth = 0, runtimeVer = 0, driverVer = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, dev));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVer));
    CUDA_CHECK(cudaDriverGetVersion(&driverVer));
    const double peakGBs = 2.0 * memClockKHz * 1e3 * (busWidth / 8.0) / 1e9;   // DDR: x2

    if (prop.warpSize != 32) {
        std::fprintf(stderr, "This program assumes warpSize == 32 (found %d).\n", prop.warpSize);
        return EXIT_FAILURE;
    }

    const std::string gpuName = prop.name;
    const std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);

    std::printf("=====================================================================\n");
    std::printf(" Parallel Reduction in CUDA  |  CUDA Lab Evaluation - Theme 1\n");
    std::printf("=====================================================================\n");
    std::printf(" GPU                    : %s\n", prop.name);
    std::printf(" Compute capability     : %s\n", cc.c_str());
    std::printf(" SMs                    : %d\n", prop.multiProcessorCount);
    std::printf(" Global memory          : %.2f GiB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf(" Shared memory / block  : %zu bytes\n", prop.sharedMemPerBlock);
    std::printf(" Max threads / block    : %d\n", prop.maxThreadsPerBlock);
    std::printf(" Warp size              : %d\n", prop.warpSize);
    std::printf(" Theoretical peak BW    : %.1f GB/s (%d-bit bus)\n", peakGBs, busWidth);
    std::printf(" CUDA runtime / driver  : %d.%d / %d.%d\n", runtimeVer / 1000, (runtimeVer % 100) / 10,
                driverVer / 1000, (driverVer % 100) / 10);
    std::printf(" Input size             : %llu floats (%.1f MiB)\n", opt.n, opt.n * 4.0 / (1024 * 1024));
    std::printf(" Timing                 : %d warm-up + %d timed runs (CUDA events), average reported\n",
                opt.warmup, opt.iters);
    std::printf("---------------------------------------------------------------------\n");

    if (!blockSizeValid(opt.block, prop, true)) {
        std::fprintf(stderr, "Invalid --block %u for this GPU.\n", opt.block);
        return EXIT_FAILURE;
    }

    // Block sizes to test, filtered by device limits.
    std::vector<unsigned int> sweepBlocks;
    for (unsigned int bs : {128u, 256u, 512u, 1024u})
        if (blockSizeValid(bs, prop, true)) sweepBlocks.push_back(bs);
    if (std::find(sweepBlocks.begin(), sweepBlocks.end(), opt.block) == sweepBlocks.end())
        sweepBlocks.push_back(opt.block);
    std::sort(sweepBlocks.begin(), sweepBlocks.end());

    // ---------------- Self-test on awkward sizes ----------------
    bool selfOk = true;
    if (opt.selftest) {
        std::printf("[1] Edge-case correctness test\n");
        selfOk = selfTest(sweepBlocks, opt.seed);
    }

    // ---------------- Host data + CPU reference ----------------
    const u64 n = opt.n;
    const size_t bytes = (size_t)n * sizeof(float);
    std::vector<float> h_in((size_t)n);
    fillDeterministic(h_in, opt.seed);

    std::printf("[2] CPU reference (std::chrono, 1 warm-up + 10 runs)\n");
    double cpuRef = cpuSum(h_in.data(), (size_t)n);                 // warm-up
    double cpuMs = 0.0;
    const int cpuRuns = 10;
    for (int r = 0; r < cpuRuns; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpuRef = cpuSum(h_in.data(), (size_t)n);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    cpuMs /= cpuRuns;
    std::printf("  CPU sum = %.6f   time = %.3f ms\n", cpuRef, cpuMs);

    // ---------------- Device buffers (allocated once, outside timing) ----------------
    float *d_in = nullptr, *bufA = nullptr, *bufB = nullptr;
    const u64 maxPartials = ceilDiv(n, 64);                        // worst case: naive pass 1, smallest block
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&bufA, (size_t)maxPartials * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bufB, (size_t)maxPartials * sizeof(float)));

    // Grid-size check: first pass has the most blocks.
    const u64 maxBlocksNeeded = ceilDiv(n, sweepBlocks.front());
    if (maxBlocksNeeded > (u64)prop.maxGridSize[0]) {
        std::fprintf(stderr, "Input too large: needs %llu blocks, device allows %d.\n",
                     maxBlocksNeeded, prop.maxGridSize[0]);
        return EXIT_FAILURE;
    }

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));   // not timed

    // ---------------- Main comparison at the chosen block size ----------------
    std::printf("[3] Kernel-only benchmark at block size %u\n", opt.block);
    GpuResult naive = benchmark(Method::Naive, d_in, n, opt.block, bufA, bufB, cpuRef, opt, evStart, evStop);
    GpuResult optim = benchmark(Method::Optimised, d_in, n, opt.block, bufA, bufB, cpuRef, opt, evStart, evStop);

    // Optional end-to-end timing (separate from kernel-only numbers):
    // H2D copy of the input + all optimised passes + D2H copy of the scalar.
    double e2eMs = 0.0;
    {
        const int e2eRuns = 10;
        const float* d_res = nullptr;
        float h = 0.0f;
        for (int r = 0; r < e2eRuns; ++r) {
            CUDA_CHECK(cudaEventRecord(evStart));
            CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
            launchReduction(Method::Optimised, d_in, n, opt.block, bufA, bufB, &d_res);
            CUDA_CHECK(cudaMemcpy(&h, d_res, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(evStop));
            CUDA_CHECK(cudaEventSynchronize(evStop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
            e2eMs += ms;
        }
        e2eMs /= e2eRuns;
    }

    std::printf("\n RESULTS TABLE  (N = %llu, block = %u, GPU = %s)\n", n, opt.block, prop.name);
    std::printf(" +---------------+-----------+-----------+----------+--------------------+-----------+--------+\n");
    std::printf(" | Method        | Time (ms) | BW (GB/s) | Speedup  | Final sum          | Rel. err  | Match  |\n");
    std::printf(" +---------------+-----------+-----------+----------+--------------------+-----------+--------+\n");
    std::printf(" | CPU (ref)     | %9.3f | %9.2f | %7.2fx | %18.4f | %9s | %-6s |\n",
                cpuMs, gbPerSec(n, cpuMs), 1.0, cpuRef, "-", "ref");
    for (const GpuResult* r : {&naive, &optim})
        std::printf(" | %-13s | %9.4f | %9.2f | %7.2fx | %18.4f | %9.2e | %-6s |\n",
                    methodName(r->method), r->avgMs, gbPerSec(n, r->avgMs), cpuMs / r->avgMs,
                    r->sum, r->relErr, r->ok ? "YES" : "NO");
    std::printf(" +---------------+-----------+-----------+----------+--------------------+-----------+--------+\n");
    std::printf("  Speedup = CPU time / GPU time.  BW = N x 4 bytes / time (effective input bandwidth;\n");
    std::printf("  partial-sum reads/writes of later passes are excluded - they add < 1%% traffic).\n");
    std::printf("  Optimised vs naive: %.2fx faster.  Passes: naive %d, optimised %d.\n",
                naive.avgMs / optim.avgMs, naive.passes, optim.passes);
    std::printf("  Peak-BW utilisation of optimised kernel: %.1f %% of %.1f GB/s\n",
                100.0 * gbPerSec(n, optim.avgMs) / peakGBs, peakGBs);
    std::printf("  Tolerance: |gpu - cpu| / |cpu| <= %.0e (float vs double; order of additions differs)\n", kRelTol);
    std::printf("  Optional end-to-end (H2D copy + optimised kernels + D2H, avg of 10, pageable memory): %.3f ms\n", e2eMs);

    // ---------------- Block-size sweep ----------------
    std::vector<GpuResult> sweep;
    if (opt.sweep) {
        std::printf("\n[4] Block-size experiment (kernel-only, avg of %d runs)\n", opt.iters);
        std::printf(" +-------+-----------------+---------------+-----------------+---------------+\n");
        std::printf(" | Block | Naive time (ms) | Naive BW GB/s | Optim time (ms) | Optim BW GB/s |\n");
        std::printf(" +-------+-----------------+---------------+-----------------+---------------+\n");
        for (unsigned int bs : sweepBlocks) {
            GpuResult a = benchmark(Method::Naive, d_in, n, bs, bufA, bufB, cpuRef, opt, evStart, evStop);
            GpuResult b = benchmark(Method::Optimised, d_in, n, bs, bufA, bufB, cpuRef, opt, evStart, evStop);
            std::printf(" | %5u | %15.4f | %13.2f | %15.4f | %13.2f |%s\n", bs, a.avgMs, gbPerSec(n, a.avgMs),
                        b.avgMs, gbPerSec(n, b.avgMs), (a.ok && b.ok) ? "" : "  <-- MISMATCH");
            sweep.push_back(a);
            sweep.push_back(b);
        }
        std::printf(" +-------+-----------------+---------------+-----------------+---------------+\n");
        for (Method m : {Method::Naive, Method::Optimised}) {
            const GpuResult* best = nullptr;
            for (const auto& r : sweep)
                if (r.method == m && (!best || r.avgMs < best->avgMs)) best = &r;
            if (best) std::printf("  Best block size for %-13s : %u (%.4f ms, %.2f GB/s)\n",
                                  methodName(m), best->block, best->avgMs, gbPerSec(n, best->avgMs));
        }
    }

    // ---------------- CSV output ----------------
    {
        std::filesystem::path p(opt.csv);
        std::error_code ec;
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
        std::ofstream f(opt.csv);
        if (!f) {
            std::fprintf(stderr, "Warning: could not write %s\n", opt.csv.c_str());
        } else {
            f << "section,gpu,compute_capability,n,block_size,method,avg_ms,min_ms,max_ms,"
                 "bandwidth_GBps,speedup_vs_cpu,final_sum,rel_error,correct\n";
            char line[512];
            std::snprintf(line, sizeof line, "main,%s,%s,%llu,-,CPU,%.6f,-,-,%.4f,1.00,%.6f,0,REF\n",
                          csvSafe(gpuName).c_str(), cc.c_str(), n, cpuMs, gbPerSec(n, cpuMs), cpuRef);
            f << line;
            auto row = [&](const char* sec, const GpuResult& r) {
                std::snprintf(line, sizeof line, "%s,%s,%s,%llu,%u,%s,%.6f,%.6f,%.6f,%.4f,%.4f,%.6f,%.3e,%s\n",
                              sec, csvSafe(gpuName).c_str(), cc.c_str(), n, r.block, methodName(r.method),
                              r.avgMs, r.minMs, r.maxMs, gbPerSec(n, r.avgMs), cpuMs / r.avgMs, r.sum,
                              r.relErr, r.ok ? "YES" : "NO");
                f << line;
            };
            row("main", naive);
            row("main", optim);
            for (const auto& r : sweep) row("block_sweep", r);
            std::printf("\n  Results written to %s\n", opt.csv.c_str());
        }
    }

    // ---------------- Cleanup ----------------
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(bufA));
    CUDA_CHECK(cudaFree(bufB));

    bool sweepOk = true;
    for (const auto& r : sweep) sweepOk = sweepOk && r.ok;
    const bool allOk = selfOk && naive.ok && optim.ok && sweepOk;
    std::printf("\n  OVERALL CORRECTNESS: %s\n", allOk ? "PASS - all GPU sums match the CPU reference"
                                                      : "FAIL - see messages above");
    CUDA_CHECK(cudaDeviceReset());
    return allOk ? EXIT_SUCCESS : EXIT_FAILURE;
}
