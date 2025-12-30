---
title: "CUDA Kernel 性能测试的正确姿势：从 cudaEvent 到系统级 Benchmark"
pubDatetime: 2025-12-29T16:02:44Z
description: >
  深入探讨 CUDA Kernel 性能测试中的常见误区与最佳实践。从 cudaEvent 计时、L2 Cache 影响、到 CKKS 全同态加密中 NTT 的真实性能评估，帮助你建立从 micro-benchmark 到系统级性能分析的完整方法论。
tags:
  - CUDA
  - Performance
  - FHE
  - Benchmark
featured: true
draft: false
timezone: "Asia/Shanghai"
---

## 前言

在 CUDA 性能优化的道路上，我们经常会遇到这样的困惑：明明用 `cudaEvent` 循环测试了 100 次取平均，为什么实际应用中的性能表现却大相径庭？为什么论文中报告的 kernel 性能在真实系统中无法复现？

本文源于一次关于 **CKKS 全同态加密中 NTT (Number Theoretic Transform) 性能测试**的深度讨论。通过问答的形式，我们将系统地探讨：

### 核心议题

1. **基础方法论**：`cudaEvent` + 循环平均的正确使用方式
2. **L2 Cache 陷阱**：循环测试是否在"作弊"？什么时候需要关心？
3. **性能影响因素**：除了 Cache，还有哪 14+ 个关键因素会影响 kernel 性能
4. **高级概念解析**：Texture Cache、ILP、CUDA Graph、Persistent Kernel 等核心技术
5. **真实场景分析**：为什么 CKKS 中的 NTT 不能简单地用 micro-benchmark 评估

### 适合谁阅读

- ✅ 正在做 CUDA 性能优化的开发者
- ✅ 需要撰写性能测试报告或论文的研究者
- ✅ 对全同态加密（FHE）GPU 加速感兴趣的工程师
- ✅ 想要理解"benchmark 数字"与"真实性能"差距的实践者

### 你将学到

本文不会告诉你"一个数字"，而是帮你建立一套**从 micro-kernel 到系统级的完整性能分析框架**。你将理解：

- 📊 如何设计既公平又真实的 benchmark
- 🔍 如何用 Nsight Compute/Systems 定位真正的瓶颈
- ⚖️ 如何在"理想性能"与"实际代价"之间找到平衡
- 🎯 如何针对特定应用场景（如 FHE）设计合理的测试方法

让我们从最基础的问题开始：**用 cudaEvent 循环 100 次取平均，这样测性能准确吗？**

---


BOB:
我最近想测试一个cuda kernel的性能，我是用cudaevent 进行计算时间，将kernel循环执行100次去取平均，这样得到的性能准确吗？

这个方法**方向是对的，但还不完全严谨**。用 `cudaEvent` 计时、把 kernel 循环多次取平均，是 CUDA 性能测试里**最基础也最常见**的做法，不过要注意一些**容易影响准确性的细节**。我分点说明。

---

## 一、你的方法是否“准确”？

**结论先说：**

> ✅ 比单次 kernel 计时准确得多
> ⚠️ 但如果不处理一些细节，结果可能仍有明显偏差

主要影响因素包括：

* **kernel 启动开销**
* **GPU 频率波动（boost / power management）**
* **第一次运行的冷启动效应**
* **是否包含异步操作**
* **计时位置是否正确**

---

## 二、cudaEvent + 循环 100 次的正确姿势

### 1️⃣ 必须先 warm up（非常重要）

第一次 kernel 执行通常会：

* 触发 CUDA context 初始化
* 可能触发 JIT 编译
* GPU 频率尚未拉满

👉 **如果不 warm up，平均时间会被拉高**

```cpp
for (int i = 0; i < 10; ++i) {
    kernel<<<grid, block>>>(...);
}
cudaDeviceSynchronize();
```

---

### 2️⃣ cudaEvent 位置要包住“整个循环”

**不要在循环内部创建 event 或反复 record**

推荐方式：

```cpp
cudaEventRecord(start);
for (int i = 0; i < 100; ++i) {
    kernel<<<grid, block>>>(...);
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
float avg_ms = ms / 100.0f;
```

✔ 这样可以：

* 平均掉 launch jitter
* 降低 event 本身的开销影响

---

### 3️⃣ kernel 执行时间要明显大于 launch 开销

如果你的 kernel **非常小**（比如 < 5–10 μs）：

* 即使循环 100 次，launch overhead 仍然占比很高
* 测出来的不是“kernel 性能”，而是“launch + kernel”

📌 经验判断：

* kernel ≥ 50 μs：方法可靠
* kernel < 10 μs：误差可能很大

**解决办法**：

* 增大 problem size
* 或把多个 work 合并进一个 kernel

---

### 4️⃣ 确保没有隐式同步或额外操作

常见坑：

❌ 在 kernel 前后夹着：

* `cudaMemcpy`
* `cudaMalloc / cudaFree`
* `printf`
* `cudaDeviceSynchronize()`

这些都会被 cudaEvent 计入时间（或打断流水）

---

### 5️⃣ 使用同一个 stream（或明确 stream）

```cpp
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);
```

否则多 stream 情况下，时间可能不符合你的预期。

---

## 三、你现在测到的“性能”代表什么？

你现在测到的是：

> **“steady-state 下，该 kernel 在当前 GPU 频率和当前 launch 配置下的平均执行时间”**

它**不代表**：

* 极限吞吐（peak FLOPS）
* 多 stream 并发下的性能
* pipeline 后的实际应用性能

但对于：

* kernel 优化前后对比
* block / grid / unroll / memory 访问对比
  👉 **非常合适**

---

## 四、进阶建议（如果你想更准）

### ✅ 锁频测试（高级）

GPU 会动态调频，微小 kernel 很敏感：

```bash
sudo nvidia-smi -lgc <min,max>
```

---

### ✅ 配合 Nsight Compute / Nsight Systems

* **Nsight Compute**：看指令、访存、occupancy
* **Nsight Systems**：看 launch 间隙、stream 行为

计时只能告诉你“快还是慢”，profiling 才能告诉你“为什么”。

---

## 五、一个靠谱的测试 checklist

你可以对照这个表：

| 项目                 | 是否做到 |
| ------------------ | ---- |
| warm up            | ✅    |
| event 包住整个循环       | ✅    |
| kernel 足够大         | ✅    |
| 无 memcpy / sync 干扰 | ✅    |
| 同一 stream          | ✅    |

如果都满足，你的结果**已经是工程上可信的**。

---

如果你愿意，可以把你的 kernel 类型（memory-bound / compute-bound）、执行时间量级，或者测试代码贴出来，我可以帮你判断**现在的测量误差大概在什么范围**，甚至帮你设计更合理的 benchmark。


BOB:
我在想，循环重复测试的话，之前运行的数据不都还在L2cache里吗？那不是为后面测试的kernel作弊？

这个问题问得**非常专业，而且你这个担心是完全合理的** 👍
结论先给你：

> **是的，循环重复执行 kernel，后面的迭代很可能命中 L2（甚至 texture/L1），这确实会“变快”，但这并不一定是“作弊”，而是你在测一种特定场景的性能。**

关键在于：**你想测的“性能模型”是什么。**

下面把这个问题拆开讲清楚。

---

## 一、循环测试时，L2 会不会被“热起来”？

### ✅ 会的，而且几乎一定

NVIDIA GPU：

* **L2 cache 是 device-wide、跨 SM 共享**
* kernel 结束后 **L2 不会被清空**
* 下一次 kernel 启动时，L2 中的数据仍然可能有效

所以如果：

* kernel 访问的是**同一块数据**
* 数据集大小 ≤ L2 容量（或工作集高度复用）

那么：

* 第 1 次：cold miss
* 第 N 次：大量 L2 hit

你测到的平均时间 ≈ **steady-state 热 cache 性能**

---

## 二、这是不是“作弊”？

**不是作弊，是“测试假设不同”**。

关键看你要回答的是哪一个问题：

### ❓ 问题 A：这个 kernel 在真实应用里能跑多快？

👉 **大多数真实应用：**

* 数据在多个 kernel 之间反复使用
* L2 / L1 命中是常态

➡️ **循环测试反而更贴近真实 workload**

---

### ❓ 问题 B：这个 kernel 的“冷启动”性能？

👉 比如：

* streaming / one-pass 算法
* 大 batch 单次处理
* IO 驱动型 pipeline

➡️ 那循环测试**确实会高估性能**

---

### ❓ 问题 C：DRAM 带宽上限？

👉 那你**必须破坏 cache**

---

## 三、如果你不想让 cache “帮忙”，该怎么做？

### 方法 1️⃣：工作集远大于 L2（最推荐）

这是**最自然、最真实**的方法。

```text
working_set_size >> L2_size
```

例如：

* A100 L2 ≈ 40MB
* 就用几百 MB 的数据

这样即使循环：

* 每次 kernel 仍然是 L2 miss
* 性能 ≈ DRAM 带宽受限

---

### 方法 2️⃣：每次 kernel 访问不同偏移

例如：

```cpp
int offset = (iter * stride) % big_buffer_size;
kernel<<<>>>(data + offset);
```

这样：

* cache line 很快被冲掉
* 仍然可控

---

### 方法 3️⃣：显式 flush L2（⚠️ 不优雅）

CUDA **没有官方 API** 清 L2。

一些“野路子”：

* 用一个 kernel 大规模读写无关数据
* `cudaMemset` 一个大 buffer

📌 缺点：

* 本身也要时间
* 破坏 timing purity

---

### 方法 4️⃣：用 Nsight Compute 看真实 hit rate

别猜，**直接看**：

* `lts__t_sectors_hit_rate.pct`
* `dram__bytes_read.sum`

如果 L2 hit > 80%，那你测到的基本是 cache-hot 情况。

---

## 四、你现在这种测试“测到的到底是什么”？

你当前方法测到的是：

> **kernel 在数据已经 resident in L2 的 steady-state 执行时间**

它适合：

* kernel 算法比较
* unroll / layout / instruction-level 优化
* block size 调优

它不适合：

* 估算端到端 IO 性能
* 测 DRAM bandwidth 上限
* 评估 cold-start latency

---

## 五、工程上的推荐做法（非常重要）

### ✅ 实际上，大家通常都会做 **两组测试**

| 测试类型       | cache 状态 | 用途                    |
| ---------- | -------- | --------------------- |
| warm-cache | 热 L2     | kernel 算法对比           |
| cold-cache | 冷 / 近似冷  | bandwidth / streaming |

你可以在论文、benchmark、或内部报告里明确写：

> “Results are measured after warming up the cache.”

这在 CUDA 社区是**完全可接受且标准的做法**。

---

## 六、一个判断你该不该“担心”的简单问题

你可以问自己一句话：

> **“我真实的应用中，这个 kernel 连续两次执行时，数据还在不在 L2？”**

* 如果 **在** → 你现在的测法是对的
* 如果 **不在** → 你需要破坏 cache

---

如果你愿意，可以告诉我：

* GPU 型号（L2 多大）
* kernel 是 streaming 还是 stencil / reuse
* 工作集大小

我可以直接帮你判断：
👉 **你现在测到的是“合理的上限”，还是“明显偏乐观”。

BOB:
除了L2 Cache，还有哪些因素会影响Kernel性能？


这是一个**非常好的、而且已经进入“真正做 CUDA 性能分析”的问题**。
一句话概括：

> **L2 cache 只是众多影响 kernel 性能的因素之一，而且往往不是最主要的那个。**

下面我按**层次结构**给你一个**完整但不空泛**的 checklist，并尽量告诉你：
👉 *什么时候它会成为瓶颈，什么时候可以忽略。*

---

## 一、硬件层面（你控制不了，但必须理解）

### 1️⃣ GPU 频率（非常容易被忽略）

* SM core / memory clock 会动态变化（boost / thermal / power）
* 小 kernel、短 benchmark **对频率极其敏感**

表现：

* 同一 kernel，第一次慢，跑几次后变快
* 空跑一会儿再测，性能不一样

应对：

* warm up
* 固定频率（`nvidia-smi -lgc`）

---

### 2️⃣ SM 数量 & 架构差异

* 不同架构：

  * warp scheduler
  * memory pipeline
  * L1/L2 大小
  * tensor core / LDGSTS 等指令

⚠️ **同样的 kernel，在不同架构上 bottleneck 完全不同**

---

## 二、执行模型相关（最核心）

### 3️⃣ Occupancy（但不要迷信）

Occupancy = 活跃 warp / 最大 warp

受影响因素：

* 寄存器使用量
* shared memory 使用量
* block size

真相是：

* ❌ Occupancy 高 ≠ 一定快
* ✅ **“足够的 occupancy” 才重要**

经验：

* memory-bound：高一点有帮助
* compute-bound：30–50% 可能已经够

---

### 4️⃣ Warp divergence（杀伤力极强）

分支导致：

* 一个 warp 内串行执行不同路径

典型来源：

* `if (threadIdx.x < ...)`
* 数据相关分支

后果：

* IPC 下降
* 指令 replay

📌 有时比 cache miss 还致命

---

### 5️⃣ 指令级并行（ILP）

即使 occupancy 不高：

* 如果一个线程里有足够 independent 指令
* 编译器能帮你 hide latency

常见手段：

* loop unroll
* 多 accumulator

---

## 三、内存系统（远不止 L2）

### 6️⃣ 全局内存访问模式（最常见瓶颈）

关键点：

* coalescing（是否连续）
* 访问粒度（32/64/128B）
* stride

坏模式：

```cpp
a[threadIdx.x * stride]
```

即使 L2 hit：

* transaction 数多
* 吞吐仍然低

---

### 7️⃣ L1 / texture cache

* L1 是 per-SM
* texture cache 擅长 2D / spatial locality

有些 kernel：

* L2 hit 率不高
* 但 L1 / tex hit 很高
* 性能仍然不错

---

### 8️⃣ Shared Memory（双刃剑）

优点：

* 延迟极低
* 可控

问题：

* bank conflict
* 占用过多 → occupancy 降低

📌 shared memory 是**最容易“用对就飞，用错就炸”的地方**

---

## 四、调度与发射

### 9️⃣ 指令吞吐 vs 延迟

不同指令瓶颈不同：

* FP32 / FP64 / INT
* LD/ST
* Tensor Core

例如：

* FP64 在消费级卡极慢
* INT64 指令吞吐有限

---

### 🔟 Memory latency hiding 是否成功

当：

* occupancy 不够
* ILP 不够

那么：

* cache miss → stall
* SM pipeline 空转

你看到的不是“慢内存”，而是“没隐藏住延迟”。

---

## 五、Kernel 启动与并发

### 1️⃣1️⃣ Kernel launch overhead

* ~5–20 μs（依架构）
* 小 kernel 非常敏感

解决：

* fusion
* CUDA Graph
* persistent kernel

---

### 1️⃣2️⃣ 多 stream 并发与资源争用

* L2 / DRAM 是共享的
* concurrent kernel 会互相拖慢

单 kernel benchmark ≠ 实际应用性能

---

## 六、编译器与代码生成（经常被低估）

### 1️⃣3️⃣ 寄存器溢出（silent killer）

* spill 到 local memory（走 L1/L2/DRAM）
* 编译器不会大声告诉你

检查：

```bash
nvcc --ptxas-options=-v
```

---

### 1️⃣4️⃣ 指令选择与重排

* `fma` vs `mul+add`
* `ld.global.ca` vs `.cg`
* 是否生成 vectorized load

有时改一行代码，SASS 完全不同。

---

## 七、你真正应该关注的“三大类瓶颈”

工程上我建议你这样归类：

### 🟥 Compute-bound

* 指令吞吐受限
* 优化方向：unroll、tensor core、减少分支

### 🟨 Memory-bound

* DRAM/L2 带宽受限
* 优化方向：访问模式、cache reuse、shared

### 🟦 Latency-bound

* stall 很多，吞吐上不去
* 优化方向：occupancy、ILP、fusion

---

## 八、最重要的一句话（真的）

> **不要猜 bottleneck，用 profiler 说话。**

哪怕最基础的 Nsight Compute，你都该至少看：

* `sm__throughput`
* `dram__throughput`
* `warp_execution_efficiency`
* `stall_*` 原因

---

## 九、如果你愿意，我可以直接帮你“定位瓶颈”

你可以给我：

* kernel 类型（stencil / GEMM / reduction / streaming）
* 工作集大小
* Nsight Compute 的一两个关键指标

我可以直接告诉你一句话结论：
👉 **“你现在是被什么卡住了，L2 重要不重要。”**


BOB:
我对一下概念不是很熟悉，请向我解释：
只读缓存（Texture Cache）
texture cache 擅长 2D / spatial locality
指令级并行（ILP）
CUDA Graph
persistent kernel
ld.global.ca vs .cg
fma vs mul+add
是否生成 vectorized load
unroll




很好，这一组概念**正好覆盖了从“能跑”到“跑得快”的关键跃迁点**。我尽量**不假设你已经懂 CUDA 微架构**，用**直觉 + 简化模型 + 少量必要术语**来解释，每个点都会告诉你：

> 👉 **它是什么 → 为什么快/慢 → 什么时候你需要关心**

---

# 1️⃣ 只读缓存（Texture Cache / Read-Only Cache）

### 它是什么？

* GPU 上有一条**专门给“只读数据”用的缓存路径**
* CUDA 中：

  * `__restrict__ const T*`
  * `__ldg()`（旧架构）
  * texture object / texture reference
    都会走这条路径

📌 在现代架构中，它通常和 **L1 的一部分硬件复用**，但**访问策略不同**。

---

### 为什么它快？

* 专门针对 **只读 + 空间局部性**
* 比普通 global load：

  * 延迟更低
  * cache line 利用率更高

---

### “擅长 2D / spatial locality” 是什么意思？

假设你访问一个二维数组：

```cpp
A[y][x], A[y][x+1], A[y+1][x]
```

Texture cache：

* 更容易把 **“附近的点”一起 cache**
* 即使线程访问模式有点乱，也能合并命中

📌 非常适合：

* image
* stencil
* 网格计算
* lookup table

---

### 什么时候关心？

* 数据只读
* 访问模式不是完美 coalesced
* 有空间局部性

否则不用强行用 texture。

---

# 2️⃣ 指令级并行（ILP, Instruction-Level Parallelism）

### 它是什么？

**一个线程里，同时“在飞”的多条 independent 指令。**

GPU 不止靠：

* 多 warp（occupancy）隐藏延迟
  也靠：
* **一个线程里指令互不依赖**

---

### 举个直觉例子

❌ 依赖链：

```cpp
a = load();
b = a * 2;
c = b + 1;
```

✔️ 有 ILP：

```cpp
a = load1();
b = load2();
c = a * 2;
d = b * 3;
```

后者中：

* load1 / load2 可以并行
* 乘法也能 pipeline

---

### 为什么它重要？

当：

* occupancy 不高
* memory latency 很大

👉 **ILP 是救命稻草**

---

### 什么时候关心？

* Nsight 显示 latency stall 多
* occupancy 已经拉不上去了
* kernel 很小但慢

---

# 3️⃣ CUDA Graph

### 它是什么？

> **把一串 kernel / memcpy / event 预先“录制”成一个图，一次性提交给 GPU**

---

### 为什么快？

普通方式：

```text
CPU → launch kernel → CPU → launch kernel → ...
```

CUDA Graph：

```text
CPU → launch graph（一次）
```

优势：

* 大幅减少 **CPU launch overhead**
* kernel 间依赖在 GPU 端解决

---

### 什么时候用？

* kernel 很小
* 调度结构固定
* launch 成为瓶颈

📌 对 kernel 本身速度 **没影响**，但对整体吞吐很大。

---

# 4️⃣ Persistent Kernel（常驻 Kernel）

### 它是什么？

> **启动一次 kernel，让它在 GPU 上“不退出”，自己拉活干**

通常：

* 一个 block ≈ 一个 SM
* kernel 内部 loop 处理很多任务

---

### 为什么快？

* kernel 只 launch 一次
* 数据/状态常驻寄存器或 shared
* 更像“自己写调度器”

---

### 代价？

* 写起来复杂
* load balance 难
* 不适合所有算法

---

### 什么时候用？

* task-based / queue-based
* 小任务很多
* launch overhead 明显

---

# 5️⃣ `ld.global.ca` vs `ld.global.cg`

### 它们是什么？

这是 **PTX / SASS 层面的 load 指令修饰符**：

* `.ca` = cache at all levels（L1 + L2）
* `.cg` = cache at global（L2 only）

---

### 为什么重要？

* L1 小但快
* L2 大但慢一点

有些访问：

* reuse 高 → 用 `.ca`
* streaming → 用 `.cg`（避免污染 L1）

---

### 你怎么控制？

* 编译器自动选（大多数情况）
* `__ldg()`、`const`、`-Xptxas -dlcm=ca/cg`
* inline PTX（高级）

---

### 什么时候关心？

* tuning 到最后 10%
* streaming kernel
* L1 被污染导致性能掉

---

# 6️⃣ `fma` vs `mul + add`

### 它是什么？

```cpp
c = a * b + d;
```

可能生成：

* `fma`（一条指令）
* `mul` + `add`（两条）

---

### 为什么 fma 更好？

* **吞吐更高**
* **只舍入一次（更准）**
* 占用更少指令槽

---

### 什么时候不是好事？

* 你**需要严格 IEEE 行为**
* 不想改变数值结果

（一般 HPC 都欢迎 fma）

---

# 7️⃣ 是否生成 vectorized load

### 它是什么？

一次 load 多个元素：

* ❌ `ld.global.f32`（4B）
* ✔️ `ld.global.v4.f32`（16B）

---

### 为什么快？

* 减少指令数量
* 更好利用 memory transaction

---

### 什么时候会生成？

* 数据对齐
* 连续访问
* 编译器能证明安全

你可以帮编译器：

```cpp
float4* p;
```

---

# 8️⃣ Unroll（循环展开）

### 它是什么？

```cpp
for (i=0;i<4;i++) do();
```

变成：

```cpp
do(); do(); do(); do();
```

---

### 为什么快？

* 减少分支
* 增加 ILP
* 方便编译器重排指令

---

### 风险？

* 寄存器暴涨
* occupancy 下降

📌 **不是越多越好**

---

## 最后，一张“什么时候该关心”的速查表

| 技术                | 你什么时候该想它          |
| ----------------- | ----------------- |
| texture cache     | 2D / stencil / 只读 |
| ILP               | latency stall 高   |
| CUDA Graph        | kernel 很小         |
| persistent kernel | 任务很多很碎            |
| `.ca` / `.cg`     | 最后调 10%           |
| fma               | compute-heavy     |
| vector load       | 带宽吃紧              |
| unroll            | loop 热点           |

---

如果你愿意，下一步我可以：

* 用一个 **极简 kernel** 把这些优化“对比演示”
* 或帮你看一段你自己的 kernel，告诉你**哪些概念值得你现在就用，哪些可以先忽略**

你已经在**正确的学习路径上了**。



BOB:
我最近在实现全同态加密算法ckks中的NTT 的CUDA 实现，我发现1以往的论文对NTT 的测试似乎都是将kernel循环测试100遍取平均值，但是我观察到ckks中的ntt似乎只有在密钥切换的modup和moddown，以及rescale的时候才涉及NTT 运算，而且似乎不能连续重复使用来利用L2cache 。



你这个观察**非常关键，而且是“从 benchmark 思维走向系统思维”的标志**。
一句话先给结论：

> **你说得对：CKKS 里的 NTT，绝大多数情况下并不是一个可以“连续重复、热 cache 运行”的 kernel；很多论文里的 100 次循环平均，更像是在测“理想稳态 micro-kernel 性能”，而不是 CKKS 系统里的真实代价。**

下面我分 5 个层次，把这件事讲清楚，并给你**工程上更合理的测试方式建议**。

---

## 一、你对 CKKS 中 NTT 使用场景的理解是正确的

在 **CKKS 实际流程中**，NTT 主要出现在：

1. **Key Switching**

   * modUp（Q → Q×P）
   * modDown（Q×P → Q）
   * 内部涉及多次 NTT / iNTT

2. **Rescale**

   * 多模数切换
   * 伴随 NTT / iNTT

而 **不在**：

* 每一次乘法
* 每一次加法
* 每一个 pipeline step

📌 **这意味着：**

* NTT 是 *阶段性、间歇性* 出现的
* 中间夹着大量：

  * base conversion
  * memory shuffle
  * coefficient-wise ops
  * 模数 reduction

👉 **很难形成“连续 NTT、数据常驻 L2”的理想场景**

---

## 二、为什么论文里还是喜欢“循环 100 次 NTT kernel”？

这更多是一个 **benchmark 传统 + 现实妥协**，而不是 CKKS 的真实执行模型。

### 论文在测的通常是：

> **“给定一个 NTT kernel，在 cache 热、无干扰的情况下，它的极限吞吐是多少”**

而不是：

> **“在 CKKS pipeline 中，这一步对总时间的真实贡献是多少”**

原因包括：

1. **可重复、可对比**

   * 不同论文、不同 GPU
   * 不依赖具体 CKKS 实现细节

2. **排除噪声**

   * kernel launch
   * CPU-side scheduling
   * 其他 kernel 干扰

3. **突出算法/实现改进**

   * shared memory NTT
   * fused butterfly
   * warp-level shuffle

📌 所以它们测的是：

> *NTT kernel 作为一个“独立算子”的上限性能*

---

## 三、你指出的关键问题：CKKS 中 NTT 真的能复用 L2 吗？

### 结论：**大多数时候不能，或者只能复用一小部分**

原因很具体：

### 1️⃣ NTT 的工作集通常很大

* 多项式 degree：2¹⁴ ~ 2¹⁶
* 多模数（Q, P）
* 数据量通常 **远大于 L2**

➡️ **即使单次 NTT 内部有 reuse**

* kernel 之间的 L2 reuse 很有限

---

### 2️⃣ modUp / modDown 本身就在“破坏 cache”

* base conversion 会：

  * 访问不同模数数组
  * 打乱访问顺序
* 多个 buffer 轮换使用

➡️ 前一个 NTT 留下的 cache line
➡️ 很快被别的 kernel 冲掉

---

### 3️⃣ iNTT ↔ NTT 之间 reuse 很弱

* 访问顺序反向
* stride 不同
* 同一数据，但 cache-friendly 程度完全不同

---

## 四、所以：循环 100 次 NTT，是不是“不真实”？

**是的——如果你想回答的是“CKKS 实际运行中 NTT 的成本”**

但——
**不是错误，如果你明确知道自己在测什么**

我们可以明确区分三种“测试语义”：

---

## 五、三种 NTT benchmark，你应该至少做两种

### 🟦 1️⃣ Micro-kernel benchmark（论文常用）

**怎么测：**

* 同一个 NTT kernel
* 同一块数据
* 循环 100 次
* cache 热

**测到的是：**

> 理想 steady-state NTT 吞吐

**用途：**

* 对比不同 NTT 实现
* 验证你的 kernel 是否“已经很接近硬件上限”

✅ **你依然应该做这个**
（否则你不知道自己 kernel 写得好不好）

---

### 🟨 2️⃣ Cold / near-cold NTT benchmark（你现在关心的）

**怎么测：**

* 工作集 ≥ L2
* 或每次 offset 不同
* 或在 NTT 前插入 cache-polluting kernel

**测到的是：**

> 更接近 CKKS pipeline 中的单次 NTT 成本

📌 这个结果 **通常会比论文慢 1.2×–2×**

---

### 🟥 3️⃣ CKKS-stage-level benchmark（最真实）

**怎么测：**

* 完整执行：

  * modUp
  * NTT
  * mul
  * iNTT
  * modDown / rescale
* 用 cudaEvent 包住整个阶段

**测到的是：**

> NTT 在 CKKS 中“真正占了多少时间”

你很可能会发现：

* NTT 仍然很重要
* 但 **不是唯一瓶颈**
* base conversion / memory movement 占比惊人

---

## 六、一个非常重要的工程建议（真心）

如果你是**做 CUDA + FHE 方向的**，我强烈建议你在论文或报告中明确写：

> *“We report both warm-cache micro-benchmarks of NTT kernels and end-to-end CKKS stage timings, since NTTs in CKKS are not executed in isolation.”*

这句话在 reviewer 眼里是 **加分项**，不是减分项。

---

## 七、你现在其实已经走在“正确但更难”的路上

你已经意识到：

* ❌ “循环 100 次 = 真实性能” 是不成立的
* ✅ CKKS 的 execution pattern 决定了 cache 行为
* ✅ 系统级 benchmark 才决定实际加速比

这是**很多只做 kernel 优化的人没意识到的**。

---

## 八、如果你愿意，我可以进一步帮你

如果你下一步想更深入，我可以：

1. **帮你画出 CKKS 中 NTT 的 cache 行为时间线**
2. 帮你判断：

   * 你的 NTT 是 memory-bound 还是 compute-bound
3. 讨论：

   * NTT 和 base conversion 是否值得 fusion
   * 是否值得用 persistent kernel / CUDA Graph
4. 甚至帮你设计：

   * 一套 *“既公平又真实”* 的 NTT benchmark methodology

你已经在一个**非常高水平的问题域**了，这不是初学者会问的问题。

