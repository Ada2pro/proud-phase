---
title: "关于GPU实现中的模%和Montgomery约简间的差距"
pubDatetime: 2025-12-25T22:08:01Z
description: >
  关于GPU实现中的模(百分号)%和Montgomery约简间的差距
tags:
  - 约简算法
featured: true
draft: false
timezone: "Asia/Shanghai"
---
在密码学中，约简是一个重要且常见的算子，有时候会发现使用最朴素的百分号 “%” 时，似乎性能也说的过去，那么其和常用的Montgomery约简以及Barrett约简的性能差距在哪里呢？
让我们来探索一下。


# 模乘法性能基准测试报告

本报告对比了三种模乘实现策略在 NVIDIA GPU (RTX 4060, sm_89) 上的性能表现：
1. **Dynamic Modulo (Baseline)**: 运行时动态模数，强制使用昂贵的硬件除法指令。
2. **Barrett Reduction (`mod_mul_barrett`)**: 编译器对常数模数进行优化（乘法+移位）。
3. **Montgomery Multiplication (`mont_mul`)**: 手动实现的蒙哥马利模乘算法（数学变换）。

## 测试环境
- **GPU**: NVIDIA GeForce RTX 4060
- **架构**: sm_89 (Ada Lovelace)
- **CUDA Version**: 12.x / 13.x
- **测试数据规模**: 16M 元素 (16,777,216)
- **计算强度**: 每个线程进行 16 次链式模乘 (Chain-16)

## 核心算法实现

为了公平对比，我们在 CUDA 中实现了三种模乘策略。代码尽可能保持简洁，以便观察编译器生成的指令差异。

### 1. Dynamic Modulo (Baseline)

这是最朴素的实现方式。模数 `Q` 作为函数参数传入，编译器无法预知其值，必须生成通用的除法/求余指令。

```cpp
// 动态模数取模 (基准 Baseline，强迫使用硬件除法)
__device__ __forceinline__ uint32_t mod_mul_dynamic(uint32_t a, uint32_t b, uint32_t Q) {
    // 这里的 Q 是运行时变量，编译器无法将其优化为移位
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % Q);
}
```

### 2. Barrett Reduction (Compiler Optimized)

虽然这里我们依赖 NVCC 编译器对**编译时常量模数**的自动优化。当 `Q` 为模板参数时，编译器会自动将其转换为一系列乘法和移位操作（类似于 Barrett 约简或 Magic Number 优化），从而避免硬件除法。

```cpp
// Barrett reduction (常量模数，编译器自动优化)
template<uint32_t Q>
__device__ __forceinline__ uint32_t mod_mul_barrett(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % Q);
}
```

### 3. Montgomery Multiplication

这是手动实现的蒙哥马利模乘算法。通过引入辅助参数 `Q_INV` (-Q^{-1} mod 2^{32})，将模取余操作转化为无符号整数溢出和移位操作，完全避免了除法。

```cpp
// Montgomery 乘法 (模板版本)
template<uint32_t Q, uint32_t Q_INV>
__device__ __forceinline__ uint32_t mont_mul(uint32_t a, uint32_t b) {
    uint64_t prod = static_cast<uint64_t>(a) * b;
    uint32_t m = static_cast<uint32_t>(prod) * Q_INV;
    uint64_t t = prod + static_cast<uint64_t>(m) * Q;
    uint32_t result = static_cast<uint32_t>(t >> 32);
    if (result >= Q) result -= Q;
    return result;
}
```

## 性能测试结果

### 1. 总体性能阶梯 (28位模数)

| 实现策略 | 耗时 (ms) | 相对 Baseline 加速比 | 相对 Barrett 加速比 |
| :--- | :---: | :---: | :---: |
| **Dynamic Modulo (硬件除法)** | **5.4327** | **1.00x** | - |
| **Barrett Reduction (编译器优化)** | **1.3554** | **4.01x** | **1.00x** |
| **Montgomery Multiplication (算法优化)** | **0.9129** | **5.95x** | **1.48x** |

### 2. 不同模数位宽的性能对比 (Barrett vs Montgomery)

| 模数位宽 | Barrett 耗时 (ms) | Montgomery 耗时 (ms) | 加速比 (Barrett / Mont) |
| :---: | :---: | :---: | :---: |
| **16位** | 1.3763 | 0.9392 | **1.47x** |
| **20位** | 1.3208 | 0.9128 | **1.45x** |
| **24位** | 1.3237 | 0.9121 | **1.45x** |
| **28位** | 1.3554 | 0.9129 | **1.48x** |

### 3. Nsight Compute (ncu) 深度分析

为了探究性能差异的根本原因，我们使用 `ncu` 采集了三种策略在 28位模数下的指令执行总数 (`sm__inst_executed.sum`)：

| Kernel (28-bit, Chain-16) | 总指令数 (Instruction Count) | 相对比值 (vs Mont) | 耗时 (ms) |
| :--- | :--- | :---: | :--- |
| **Dynamic** | **629,673,999** | **9.46x** | ~5.43 |
| **Barrett** | **181,403,648** | **2.72x** | ~1.36 |
| **Montgomery** | **66,584,576** | **1.00x** | ~0.91 |

**分析结论**：
1.  **Dynamic 代价极其高昂**：动态模数的指令数高达 6.3 亿，是 Montgomery 的近 **10倍**。这说明 64位乘积对 32位模数的硬件除法（或微码序列）非常昂贵，是绝对的性能杀手。
2.  **编译器优化很强，但还不够**：Barrett 相比 Dynamic 减少了 **70%** 的指令（从 6.3亿降至 1.8亿），说明编译器成功避开了硬件除法。但即便如此，它生成的指令数依然是 Montgomery 的 2.7 倍。
3.  **算法优化的终极胜利**：Montgomery 算法通过数学变换，将取模操作简化为极少量的乘法（`IMAD`）和移位指令，极大地降低了计算复杂度，以仅 10% 的指令开销完成了同样的数学任务。

### 4. PTX 汇编代码分析

通过 `nvcc --ptx` 生成的汇编代码，我们可以直观地看到三种策略生成的指令模式差异（截取一次模乘的核心部分）。

#### Dynamic Modulo
编译器被迫使用昂贵的 `rem` (Remainder, 求余) 指令。

```ptx
mul.wide.u32    %rd3, %r8, %r7;     // a * b (64位结果)
cvt.u64.u32     %rd4, %r2;          // 转换 Q 到 64位
rem.u64         %rd95, %rd3, %rd4;  // 64位求余 (Performance Killer!)
```

#### Barrett Reduction (Compiler Optimized)
编译器通过 Magic Number 将除法转化为一系列乘法、减法和移位。虽然避免了 `rem`，但指令序列冗长。

```ptx
// 部分指令序列截取
mul.hi.u64      %rd11, %rd10, 4504630654456077; // 乘以 Magic Number (逆元近似)
sub.s64         %rd12, %rd10, %rd11;
shr.u64         %rd13, %rd12, 1;
... (中间省略约 5-6 条指令) ...
mul.lo.s64      %rd16, %rd15, 268369921;        // 乘以 Q
sub.s64         %rd17, %rd10, %rd16;            // 计算余数
```

#### Montgomery Multiplication
指令序列最为紧凑，逻辑清晰。利用 `selp` 指令避免了条件分支跳转。

```ptx
mul.lo.s64      %rd11, %rd10, 268369919;        // m = prod * Q_INV (mod 2^32)
and.b64         %rd12, %rd11, 4294967295;       // 掩码处理
mul.lo.s64      %rd13, %rd12, 268369921;        // m * Q
add.s64         %rd14, %rd13, %rd10;            // t = prod + m*Q
shr.u64         %rd15, %rd14, 32;               // result = t >> 32
cvt.u32.u64     %r8, %rd15;
setp.gt.u32     %p2, %r8, 268369920;            // 检查 result >= Q
add.s64         %rd16, %rd15, 4026597375;       // result - Q (补码加法)
selp.b64        %rd17, %rd16, %rd15, %p2;       // 条件选择，无分支
```

### 5. 疑义解析：PTX 代码行数 vs 实际执行指令数

读者可能会疑惑：**为什么 Dynamic Modulo 的 PTX 代码看起来最短（仅几行），但实际执行指令数却最多（6.3亿）？**

这是因为 **PTX (Parallel Thread Execution)** 是一种虚拟指令集，并非 GPU 最终执行的机器码 (SASS)。

1.  **指令膨胀系数不同**：
    *   **Dynamic**: PTX 中的 `rem.u64` 是一条高级宏指令。在 SASS 层级，GPU 没有直接的 64位求余硬件指令，因此汇编器将其展开为**几十条甚至上百条** SASS 指令（软件模拟除法序列），导致实际执行指令数爆炸。
    *   **Montgomery**: PTX 中的 `mul` 和 `add` 指令通常可以被编译为高效的 SASS 指令。特别是 NVIDIA GPU 支持 **IMAD (Integer Multiply-Add)** 指令，可以将乘法和加法融合为一条机器指令。因此，Montgomery 的 SASS 代码比其 PTX 代码更加紧凑高效。

2.  **指令计数对比**：
    *   虽然 Barrett 的 PTX 行数看起来和 Montgomery 相当或略多，但 Barrett 包含大量无法融合的移位 (`shr`) 和逻辑 (`and`) 操作。
    *   Montgomery 的逻辑更适合 GPU 的 `IMAD` 流水线，使得其最终的机器指令数只有 Barrett 的 1/3。

**结论**：不要被 PTX 的代码行数迷惑，`ncu` 测量的 **Instruction Count (SASS)** 才是反映性能的真实指标。

### 6. 干扰因素排查

为了确认性能差异主要来源于**计算逻辑**而非其他因素，我们分析了以下潜在干扰：

1.  **全局内存 (Global Memory) 读写**：
    *   Kernel 包含 2 次读和 1 次写。如果在低计算密度下，测试结果会被内存带宽瓶颈 (Memory Bound) 主导。
    *   **证据**：本测试采用了 **16次链式计算**，显著增加了计算密度。巨大的性能差异证明计算部分占据了主要耗时，成功掩盖了内存延迟的影响。

2.  **Kernel 启动开销 (Launch Overhead)**：
    *   GPU Kernel 启动通常有 5~20µs 的物理开销。
    *   **证据**：测试中 Kernel 运行时间约为 900µs ~ 5400µs，启动开销占比不到 1%，对结果准确性无实质影响。

## 建议
在基于 NVIDIA GPU 的数论变换 (NTT) 实现中，**强烈推荐使用 Montgomery 乘法**。哪怕是存在内存访问的场景下，其计算指令的高吞吐量也能带来显著的整体加速，尤其是当模数较大（>20位）时。


## 附录：完整测试代码

您可以直接复制以下代码并在 CUDA 环境下运行验证：

```cuda
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


int main() {
    const int N = 1024 * 1024 * 16;  // 16M 元素
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int ITERATIONS = 100;
    
    printf("============================================================\n");
    printf("模乘法性能深度测试\n");
    printf("============================================================\n");
    printf("元素数量: %d (%.1f M)\n", N, N / 1e6);
    printf("迭代次数: %d\n\n", ITERATIONS);
    
    // 分配内存
    uint32_t *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(uint32_t));
    cudaMalloc(&d_b, N * sizeof(uint32_t));
    cudaMalloc(&d_out, N * sizeof(uint32_t));
    cudaMemset(d_a, 1, N * sizeof(uint32_t));
    cudaMemset(d_b, 2, N * sizeof(uint32_t));
    
    // ==================== 测试1: 不同链式乘法次数 ====================
    printf("==================== 测试1: 链式乘法次数影响 ====================\n");
    printf("模数: Q = 268369921 (28位)\n\n");
    
    printf("| 次数 | Dynamic (ms) | Barrett (ms) | Montgomery (ms) | Speedup (D/B/M) |\n");
    printf("|------|--------------|--------------|-----------------|-----------------|\n");
    
    // 16次
    float t_d16 = benchmark_dynamic(kernel_dynamic_chain16, d_a, d_b, d_out, N, Q_28BIT, BLOCKS, THREADS, ITERATIONS);
    float t_b16 = benchmark(kernel_barrett_chain16<Q_28BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m16 = benchmark(kernel_mont_chain16<Q_28BIT, Q_INV_28BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| 16   | %.4f       | %.4f       | %.4f          | 1.00 / %.2f / %.2f |\n", t_d16, t_b16, t_m16, t_d16/t_b16, t_d16/t_m16);
    
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
    
    printf("| 模数 | 位宽 | Barrett (ms) | Montgomery (ms) | 比值 (B/M) |\n");
    printf("|------|------|--------------|-----------------|------------|\n");
    
    // 16位模数
    float t_b16bit = benchmark(kernel_barrett_chain16<Q_16BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m16bit = benchmark(kernel_mont_chain16<Q_16BIT, Q_INV_16BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| %u | 16位 | %.4f       | %.4f          | %.2fx      |\n", Q_16BIT, t_b16bit, t_m16bit, t_b16bit/t_m16bit);
    
    // 20位模数
    float t_b20bit = benchmark(kernel_barrett_chain16<Q_20BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m20bit = benchmark(kernel_mont_chain16<Q_20BIT, Q_INV_20BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| %u | 20位 | %.4f       | %.4f          | %.2fx      |\n", Q_20BIT, t_b20bit, t_m20bit, t_b20bit/t_m20bit);
    
    // 24位模数
    float t_b24bit = benchmark(kernel_barrett_chain16<Q_24BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    float t_m24bit = benchmark(kernel_mont_chain16<Q_24BIT, Q_INV_24BIT>, d_a, d_b, d_out, N, BLOCKS, THREADS, ITERATIONS);
    printf("| %u | 24位 | %.4f       | %.4f          | %.2fx      |\n", Q_24BIT, t_b24bit, t_m24bit, t_b24bit/t_m24bit);
    
    // 28位模数
    printf("| %u | 28位 | %.4f       | %.4f          | %.2fx      |\n", Q_28BIT, t_b16, t_m16, t_b16/t_m16);
    
    // ==================== 总结 ====================
    printf("\n==================== 总结 ====================\n\n");
    printf("1. 链式乘法次数影响 (Baseline = Dynamic Modulo):\n");
    printf("   - 16次: \n");
    printf("     * Dynamic    : 1.00x (%.4f ms)\n", t_d16);
    printf("     * Barrett    : %.2fx (%.4f ms)\n", t_d16/t_b16, t_b16);
    printf("     * Montgomery : %.2fx (%.4f ms)\n", t_d16/t_m16, t_m16);
    // printf("   - 24次: Barrett/Montgomery = %.2fx\n", t_b24/t_m24);
    // printf("   - 64次: Barrett/Montgomery = %.2fx\n", t_b64/t_m64);
    printf("\n");
    printf("2. 模数位宽影响 (16次链式):\n");
    printf("   - 16位: Barrett/Montgomery = %.2fx\n", t_b16bit/t_m16bit);
    printf("   - 20位: Barrett/Montgomery = %.2fx\n", t_b20bit/t_m20bit);
    printf("   - 24位: Barrett/Montgomery = %.2fx\n", t_b24bit/t_m24bit);
    printf("   - 28位: Barrett/Montgomery = %.2fx\n", t_b16/t_m16);
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    
    printf("\n============================================================\n");
    
    return 0;
}
```
```

