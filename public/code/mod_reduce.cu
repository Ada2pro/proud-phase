// 测试 mod_mul 函数的 PTX 代码生成和性能对比
// 编译命令: nvcc -O3 -arch=sm_80 mul_mod.cu -o mul_mod
// 生成PTX: nvcc -O3 -arch=sm_80 --ptx mul_mod.cu -o mul_mod.ptx

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// ============================================================================
// 不同位宽的模数定义
// ============================================================================
#define Q_28BIT 268369921u   // 28位模数 (原始)
#define Q_24BIT 16769023u    // 24位模数
#define Q_20BIT 1048573u     // 20位模数
#define Q_16BIT 65521u       // 16位模数

// 对应的 -Q^(-1) mod 2^32 (Montgomery 参数)
#define Q_INV_28BIT 268369919u
#define Q_INV_24BIT 83877889u
#define Q_INV_20BIT 2386209451u
#define Q_INV_16BIT 839905007u

// ============================================================================
// Barrett reduction (常量模数，编译器自动优化)
// ============================================================================
template<uint32_t Q>
__device__ __forceinline__ uint32_t mod_mul_barrett(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % Q);
}

// ============================================================================
// 动态模数取模 (基准 Baseline，强迫使用硬件除法)
// ============================================================================
__device__ __forceinline__ uint32_t mod_mul_dynamic(uint32_t a, uint32_t b, uint32_t Q) {
    // 这里的 Q 是运行时变量，编译器无法将其优化为移位
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % Q);
}

// ============================================================================
// Montgomery 乘法 (模板版本)
// ============================================================================
template<uint32_t Q, uint32_t Q_INV>
__device__ __forceinline__ uint32_t mont_mul(uint32_t a, uint32_t b) {
    uint64_t prod = static_cast<uint64_t>(a) * b;
    uint32_t m = static_cast<uint32_t>(prod) * Q_INV;
    uint64_t t = prod + static_cast<uint64_t>(m) * Q;
    uint32_t result = static_cast<uint32_t>(t >> 32);
    if (result >= Q) result -= Q;
    return result;
}

// ============================================================================
// Shoup 约简 (预计算优化的 Barrett 变种)
// ============================================================================
template<uint32_t Q>
__device__ __forceinline__ uint32_t shoup_mul(uint32_t a, uint32_t b, uint32_t b_shoup) {
    // b_shoup = floor(b * 2^32 / Q) 是预计算的
    uint64_t prod = static_cast<uint64_t>(a) * b;
    uint64_t quot = (static_cast<uint64_t>(a) * b_shoup) >> 32;
    uint32_t result = static_cast<uint32_t>(prod) - static_cast<uint32_t>(quot) * Q;
    if (result >= Q) result -= Q;
    return result;
}

// Shoup 参数预计算函数 (CPU 端)
inline uint32_t compute_shoup_param(uint32_t b, uint32_t Q) {
    return static_cast<uint32_t>((static_cast<uint64_t>(b) << 32) / Q);
}

// ============================================================================
// 链式乘法 Kernel - 不同次数
// ============================================================================

// 16次链式乘法
__global__ void kernel_dynamic_chain16(uint32_t *a, uint32_t *b, uint32_t *out, int n, uint32_t Q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t x = a[idx], y = b[idx];
        x = mod_mul_dynamic(x, y, Q); x = mod_mul_dynamic(x, y, Q);
        x = mod_mul_dynamic(x, y, Q); x = mod_mul_dynamic(x, y, Q);
        x = mod_mul_dynamic(x, y, Q); x = mod_mul_dynamic(x, y, Q);
        x = mod_mul_dynamic(x, y, Q); x = mod_mul_dynamic(x, y, Q);
        x = mod_mul_dynamic(x, y, Q); x = mod_mul_dynamic(x, y, Q);
        x = mod_mul_dynamic(x, y, Q); x = mod_mul_dynamic(x, y, Q);
        x = mod_mul_dynamic(x, y, Q); x = mod_mul_dynamic(x, y, Q);
        x = mod_mul_dynamic(x, y, Q); x = mod_mul_dynamic(x, y, Q);
        out[idx] = x;
    }
}

template<uint32_t Q>
__global__ void kernel_barrett_chain16(uint32_t *a, uint32_t *b, uint32_t *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t x = a[idx], y = b[idx];
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        out[idx] = x;
    }
}

template<uint32_t Q, uint32_t Q_INV>
__global__ void kernel_mont_chain16(uint32_t *a, uint32_t *b, uint32_t *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t x = a[idx], y = b[idx];
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        out[idx] = x;
    }
}

template<uint32_t Q>
__global__ void kernel_shoup_chain16(uint32_t *a, uint32_t *b, uint32_t *b_shoup, uint32_t *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t x = a[idx], y = b[idx], y_s = b_shoup[idx];
        x = shoup_mul<Q>(x, y, y_s); x = shoup_mul<Q>(x, y, y_s);
        x = shoup_mul<Q>(x, y, y_s); x = shoup_mul<Q>(x, y, y_s);
        x = shoup_mul<Q>(x, y, y_s); x = shoup_mul<Q>(x, y, y_s);
        x = shoup_mul<Q>(x, y, y_s); x = shoup_mul<Q>(x, y, y_s);
        x = shoup_mul<Q>(x, y, y_s); x = shoup_mul<Q>(x, y, y_s);
        x = shoup_mul<Q>(x, y, y_s); x = shoup_mul<Q>(x, y, y_s);
        x = shoup_mul<Q>(x, y, y_s); x = shoup_mul<Q>(x, y, y_s);
        x = shoup_mul<Q>(x, y, y_s); x = shoup_mul<Q>(x, y, y_s);
        out[idx] = x;
    }
}

// 24次链式乘法 (手动展开 - 暂时注释掉)
/*
template<uint32_t Q>
__global__ void kernel_barrett_chain24(uint32_t *a, uint32_t *b, uint32_t *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t x = a[idx], y = b[idx];
        // 手动展开 24 次
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);

        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        
        out[idx] = x;
    }
}

template<uint32_t Q, uint32_t Q_INV>
__global__ void kernel_mont_chain24(uint32_t *a, uint32_t *b, uint32_t *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t x = a[idx], y = b[idx];
        // 手动展开 24 次
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);

        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);

        out[idx] = x;
    }
}
*/

// 64次链式乘法 (暂时注释掉以加速编译)
/*
template<uint32_t Q>
__global__ void kernel_barrett_chain64(uint32_t *a, uint32_t *b, uint32_t *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t x = a[idx], y = b[idx];
        for (int round = 0; round < 4; round++) {
            x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
            x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
            x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
            x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
            x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
            x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
            x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
            x = mod_mul_barrett<Q>(x, y); x = mod_mul_barrett<Q>(x, y);
        }
        out[idx] = x;
    }
}

template<uint32_t Q, uint32_t Q_INV>
__global__ void kernel_mont_chain64(uint32_t *a, uint32_t *b, uint32_t *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t x = a[idx], y = b[idx];
        for (int round = 0; round < 4; round++) {
            x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
            x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
            x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
            x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
            x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
            x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
            x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
            x = mont_mul<Q, Q_INV>(x, y); x = mont_mul<Q, Q_INV>(x, y);
        }
        out[idx] = x;
    }
}
*/

// ============================================================================
// 性能测试函数
// ============================================================================
template<typename KernelFunc>
float benchmark(KernelFunc kernel, uint32_t *d_a, uint32_t *d_b, uint32_t *d_out, 
                int n, int blocks, int threads, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}

// 专门用于动态模数 Kernel 的测试函数
template<typename KernelFunc>
float benchmark_dynamic(KernelFunc kernel, uint32_t *d_a, uint32_t *d_b, uint32_t *d_out, 
                        int n, uint32_t Q, int blocks, int threads, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    kernel<<<blocks, threads>>>(d_a, d_b, d_out, n, Q);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads>>>(d_a, d_b, d_out, n, Q);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}

// 专门用于 Shoup Kernel 的测试函数（需要额外的 b_shoup 参数）
template<typename KernelFunc>
float benchmark_shoup(KernelFunc kernel, uint32_t *d_a, uint32_t *d_b, uint32_t *d_b_shoup,
                      uint32_t *d_out, int n, int blocks, int threads, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    kernel<<<blocks, threads>>>(d_a, d_b, d_b_shoup, d_out, n);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads>>>(d_a, d_b, d_b_shoup, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}


int main() {
    const int N = 1024 * 1024 * 16;  // 16M 元素
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int ITERATIONS = 100;
    
    printf("============================================================\n");
    printf("模乘法性能深度测试 (含 Shoup 算法)\n");
    printf("============================================================\n");
    printf("元素数量: %d (%.1f M)\n", N, N / 1e6);
    printf("迭代次数: %d\n\n", ITERATIONS);
    
    // 分配内存
    uint32_t *d_a, *d_b, *d_b_shoup, *d_out;
    cudaMalloc(&d_a, N * sizeof(uint32_t));
    cudaMalloc(&d_b, N * sizeof(uint32_t));
    cudaMalloc(&d_b_shoup, N * sizeof(uint32_t));
    cudaMalloc(&d_out, N * sizeof(uint32_t));
    cudaMemset(d_a, 1, N * sizeof(uint32_t));
    cudaMemset(d_b, 2, N * sizeof(uint32_t));
    
    // 预计算 Shoup 参数（在 CPU 上）
    uint32_t *h_b = new uint32_t[N];
    uint32_t *h_b_shoup = new uint32_t[N];
    for (int i = 0; i < N; i++) {
        h_b[i] = 2;  // 与 d_b 相同
        h_b_shoup[i] = compute_shoup_param(h_b[i], Q_28BIT);
    }
    cudaMemcpy(d_b_shoup, h_b_shoup, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    delete[] h_b;
    delete[] h_b_shoup;
    
    // ==================== 测试1: 不同链式乘法次数 ====================
    printf("==================== 测试1: 链式乘法次数影响 ====================\n");
    printf("模数: Q = 268369921 (28位)\n\n");
    
    printf("| 次数 | Dynamic (ms) | Barrett (ms) | Shoup (ms) | Montgomery (ms) | Speedup (D/B/S/M) |\n");
    printf("|------|--------------|--------------|------------|-----------------|-------------------|\n");
    
    // 16次
    float t_d16 = benchmark_dynamic(kernel_dynamic_chain16, d_a, d_b, d_out, N, Q_28BIT, BLOCKS, THREADS, ITERATIONS);
    float t_b16 = benchmark(kernel_barrett_chain16<Q_28BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_s16 = benchmark_shoup(kernel_shoup_chain16<Q_28BIT>, d_a, d_b, d_b_shoup, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m16 = benchmark(kernel_mont_chain16<Q_28BIT, Q_INV_28BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| 16   | %.4f       | %.4f       | %.4f     | %.4f          | 1.00 / %.2f / %.2f / %.2f |\n", 
           t_d16, t_b16, t_s16, t_m16, t_d16/t_b16, t_d16/t_s16, t_d16/t_m16);
    
    // 24次

    /*
    float t_b24 = benchmark(kernel_barrett_chain24<Q_28BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m24 = benchmark(kernel_mont_chain24<Q_28BIT, Q_INV_28BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| 24   | %.4f       | %.4f          | %.2fx      |\n", t_b24, t_m24, t_b24/t_m24);
    */
    
    // 64次
    /*
    float t_b64 = benchmark(kernel_barrett_chain64<Q_28BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m64 = benchmark(kernel_mont_chain64<Q_28BIT, Q_INV_28BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| 64   | %.4f       | %.4f          | %.2fx      |\n", t_b64, t_m64, t_b64/t_m64);
    */
    
    // ==================== 测试2: 不同模数位宽 ====================
    printf("\n==================== 测试2: 模数位宽影响 (16次链式) ====================\n\n");
    
    printf("| 模数 | 位宽 | Barrett (ms) | Shoup (ms) | Montgomery (ms) | 比值 (B/S/M) |\n");
    printf("|------|------|--------------|------------|-----------------|---------------|\n");
    
    // 预计算不同模数的 Shoup 参数
    uint32_t *h_b_tmp = new uint32_t[N];
    uint32_t *h_b_shoup_tmp = new uint32_t[N];
    
    // 16位模数
    for (int i = 0; i < N; i++) h_b_shoup_tmp[i] = compute_shoup_param(2, Q_16BIT);
    cudaMemcpy(d_b_shoup, h_b_shoup_tmp, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    float t_b16bit = benchmark(kernel_barrett_chain16<Q_16BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_s16bit = benchmark_shoup(kernel_shoup_chain16<Q_16BIT>, d_a, d_b, d_b_shoup, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m16bit = benchmark(kernel_mont_chain16<Q_16BIT, Q_INV_16BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| %u | 16位 | %.4f       | %.4f     | %.4f          | %.2f / %.2f / 1.00 |\n", 
           Q_16BIT, t_b16bit, t_s16bit, t_m16bit, t_b16bit/t_m16bit, t_s16bit/t_m16bit);
    
    // 20位模数
    for (int i = 0; i < N; i++) h_b_shoup_tmp[i] = compute_shoup_param(2, Q_20BIT);
    cudaMemcpy(d_b_shoup, h_b_shoup_tmp, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    float t_b20bit = benchmark(kernel_barrett_chain16<Q_20BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_s20bit = benchmark_shoup(kernel_shoup_chain16<Q_20BIT>, d_a, d_b, d_b_shoup, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m20bit = benchmark(kernel_mont_chain16<Q_20BIT, Q_INV_20BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| %u | 20位 | %.4f       | %.4f     | %.4f          | %.2f / %.2f / 1.00 |\n", 
           Q_20BIT, t_b20bit, t_s20bit, t_m20bit, t_b20bit/t_m20bit, t_s20bit/t_m20bit);
    
    // 24位模数
    for (int i = 0; i < N; i++) h_b_shoup_tmp[i] = compute_shoup_param(2, Q_24BIT);
    cudaMemcpy(d_b_shoup, h_b_shoup_tmp, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    float t_b24bit = benchmark(kernel_barrett_chain16<Q_24BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_s24bit = benchmark_shoup(kernel_shoup_chain16<Q_24BIT>, d_a, d_b, d_b_shoup, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m24bit = benchmark(kernel_mont_chain16<Q_24BIT, Q_INV_24BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| %u | 24位 | %.4f       | %.4f     | %.4f          | %.2f / %.2f / 1.00 |\n", 
           Q_24BIT, t_b24bit, t_s24bit, t_m24bit, t_b24bit/t_m24bit, t_s24bit/t_m24bit);
    
    // 28位模数
    printf("| %u | 28位 | %.4f       | %.4f     | %.4f          | %.2f / %.2f / 1.00 |\n", 
           Q_28BIT, t_b16, t_s16, t_m16, t_b16/t_m16, t_s16/t_m16);
    
    delete[] h_b_tmp;
    delete[] h_b_shoup_tmp;
    
    // ==================== 总结 ====================
    printf("\n==================== 总结 ====================\n\n");
    printf("1. 链式乘法次数影响 (Baseline = Dynamic Modulo):\n");
    printf("   - 16次: \n");
    printf("     * Dynamic    : 1.00x (%.4f ms)\n", t_d16);
    printf("     * Barrett    : %.2fx (%.4f ms)\n", t_d16/t_b16, t_b16);
    printf("     * Shoup      : %.2fx (%.4f ms)\n", t_d16/t_s16, t_s16);
    printf("     * Montgomery : %.2fx (%.4f ms)\n", t_d16/t_m16, t_m16);
    printf("\n");
    printf("2. 模数位宽影响 (16次链式，相对 Montgomery):\n");
    printf("   - 16位: Barrett/Shoup/Montgomery = %.2fx / %.2fx / 1.00x\n", t_b16bit/t_m16bit, t_s16bit/t_m16bit);
    printf("   - 20位: Barrett/Shoup/Montgomery = %.2fx / %.2fx / 1.00x\n", t_b20bit/t_m20bit, t_s20bit/t_m20bit);
    printf("   - 24位: Barrett/Shoup/Montgomery = %.2fx / %.2fx / 1.00x\n", t_b24bit/t_m24bit, t_s24bit/t_m24bit);
    printf("   - 28位: Barrett/Shoup/Montgomery = %.2fx / %.2fx / 1.00x\n", t_b16/t_m16, t_s16/t_m16);
    printf("\n");
    printf("3. Shoup vs Barrett (28位模数):\n");
    printf("   - Shoup 相对 Barrett 加速: %.2fx\n", t_b16/t_s16);
    printf("   - 但仍比 Montgomery 慢: %.2fx\n", t_s16/t_m16);
    printf("   - 内存开销: Shoup 需要 2x 存储 (原值 + Shoup 参数)\n");
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b_shoup);
    cudaFree(d_out);
    
    printf("\n============================================================\n");
    
    return 0;
}
