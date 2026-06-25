---
title: "Blackwell版本GPU简介"
pubDatetime: 2026-06-24T10:00:00Z
description: >
  Blackwell版本GPU简介，包括5090、B100等
tags:
  - GPU
featured: false
draft: false
timezone: "Asia/Shanghai"
---

---

## 一.概述

blackwell架构下的nvidia gpu显卡相较于上一代生产级显卡的Hopper架构新增了一些特性以及性能改进，主要包括：

1. 增加了TMEM，Tensor Memory，专为Tensor core准备的on chip的存储空间，在B300上，每个SM有256KB，大小和SMEM一样大。但是TMEM在5090\RTX Pro 6000 Blackwell等消费级显卡上不支持。
2. 新增加了NVFP4数据类型的计算
3. tcgen05指令，仅在B100等生产级的显卡上支持。
4. 在B300上的Special Function Units (SFUs)计算吞吐量加强了很多，方便attention的计算速度。
5. blackwell引入行业首创的TEE-I/O功能

另外，flashattention4也只在B100这种卡上支持，5090、RTX Pro 6000 Blackwell并不能进行原生实现运行。

---

## 二.tcgen05

1. 通过 tcgen05.ld（加载） 指令将数据从共享内存加载到 tmem 中。

2. 执行 tcgen05.mma（矩阵乘加） 指令，让 Tensor Core 直接从 tmem 中读取数据进行计算。

3. 通过 tcgen05.st（存储） 指令将计算结果从 tmem 写回。

这种设计让 Tensor Core 能更高效地获取数据，从而大幅提升计算效率。此外，tmem 是线程束级别（Warp-level） 的，一个线程束内的所有线程共享同一块 tmem 空间，这简化了线程间的数据同步。


---

## 三.Security and reliability
看到一个安全相关的点：

> Confidential computing and secure AI: Secure and performant protection for sensitive AI models and data, extending hardware-based Trusted Execution Environment (TEE) to GPUs with industry-first TEE-I/O capabilities in the Blackwell architecture and inline NVLink protection for near-identical throughput when compared to unencrypted modes.

翻译就是：

保密计算与安全AI：为敏感的AI模型和数据提供安全且高效的保护，通过Blackwell架构引入行业首创的TEE-I/O功能，将基于硬件的可信执行环境（TEE）扩展至GPU，并在与不解密模式相比近似的数据吞吐量下提供内联NVLink保护。

也就是说，在TEE下能达到与明文下的性能差不多，

<!-- 后续会出一篇文章详细介绍这个东西。 -->

我查阅了一些资料搜集如下：

### 1.Hopper

在https://forums.developer.nvidia.com/t/security-issues-in-protected-pcie-mode/323169：

楼主问的是：

如果有人能进行物理攻击，比如接触服务器主板、NVLink/NVSwitch 线路、PCB 走线，那 NVLink/NVSwitch 上传输的数据会不会被窃取？

NVIDIA 员工的回答核心是：

> 理论上是的，存在被泄露的可能。
> 因为在这个模式下，NVLink/NVSwitch 上的数据没有加密。

但他又补充说，现实中攻击难度很高：

- NVLink 是 NVIDIA 私有协议
  攻击者很难买到像 PCIe 分析仪那样成熟的协议分析工具，想监听和解码 NVLink 数据不容易。
- 需要物理改板或者插入监听设备
  不是插根线就能监听，可能要改 PCB 走线、插入探针或中间设备。
- 监听可能破坏信号完整性
  NVLink 是高速互联，稍微改动线路就可能导致性能下降、链路错误、数据损坏，从而被用户发现。

最后他说：

> PPCIe 本来就是在性能和安全之间做了明确取舍。

也就是说，PPCIe 模式为了性能，没有对 NVLink/NVSwitch 做加密保护。

### 2.Blackwell(至少对于B300等)

在：https://forums.developer.nvidia.com/t/how-does-blackwell-support-high-performace-nvlink-encryption/337614：

这个帖子在说：Blackwell 开始，NVLink 加密不再像 Hopper 那样靠“绕路/软件/通用安全路径”付出很大代价，而是在 NVLink 数据路径上有专门的加解密硬件，所以可以接近线速加密。

帖子里的核心信息有三句：

第一，提问者说 NVIDIA Blackwell brief 里写了：

> encrypted NVLink P2P data transfer is supported without compromising performance

也就是 Blackwell 支持加密的 NVLink P2P 传输，而且声称不明显牺牲性能。

第二，NVIDIA 员工回答：

> Hopper 的 Protected PCIe mode 不支持 NVLink 加密；Blackwell 上 NVLink 有 cryptography hardware，可以在线速或接近线速进行加解密。

这句话最关键。意思是 Blackwell 的 NVLink 链路本身集成了专用加密/解密硬件，不需要把数据搬到别的慢路径处理，所以性能不会明显下降。

第三，后面有人问 B200 的 NVLink encryption 什么时候可用，NVIDIA 员工说会在后续 release 里开启；再往后有人确认 590 driver 上 8-GPU B200 系统已经可以工作，并且有 full NVLink encryption。

因此有

#### Hopper PPCIe 模式：

- PCIe 受保护
- NVLink/NVSwitch 不加密
- 所以 GPU-GPU 传输在强物理攻击下理论上可能被窃听

#### Blackwell CC 模式：

- NVLink/NVSwitch 可以支持加密
- 并且有专门硬件做加解密
- 所以目标是接近线速，不明显影响性能

一些其它参考文件：

- https://docs.nvidia.com/595trd1-trusted-computing-solutions-release-notes.pdf
- https://docs.nvidia.com/cc-deployment-guide-tdx.pdf
- https://arxiv.org/html/2606.23969v1

其中这个arxiv论文里的摘要：

> GPU Confidential Computing (GPU-CC) now preserves GPU-local performance: on NVIDIA B300, BF16 matmul runs at 0.998x of non-confidential performance. Yet LLM serving under Intel TDX plus GPU-CC still loses 13-27% of throughput, and KV-cache restore latency can more than double. This paper studies that gap on two Blackwell platforms, RTX Pro 6000 and B300 HGX, and identifies its dominant cause: the confidential VM-GPU bridge, not GPU compute.
>
> We find that GPU-CC turns host/device movement into a serialized, high-setup-cost channel. Secure copies do not gain CUDA-stream concurrency within a context, asynchronous transfers block at the runtime boundary, and small crossings pay a fixed toll. This violates the assumptions of modern inference runtimes, where DMA is expected to be cheap, concurrent, and asynchronous. In vLLM dense decode, the gap closes around 44x-slower small alloc-and-copy operations; targeted patches reject alternative explanations. A scheduling flag recovers 57% of the gap, while a worker-thread drain recovers up to 92% in qualified high-concurrency runs. The same bridge model explains a +131% KV-restore penalty and a 34x model-load slowdown.
>
> Blackwell also changes the confidential tenancy unit. We qualify confidential multi-GPU NVSwitch tenants on B300, including 510 GB/s NVLink P2P inside a CVM and concurrent isolated tenants, and identify the remaining fabric-attestation gap for production confidential AI platforms.

#### 论文里的性能损失

重点看：“loses 13-27% of throughput”、“a +131% KV-restore penalty and a 34x model-load slowdown.”。

这里性能损失主要体现在数据搬运上，而不是计算能力本身，论文分析的大致原因如下：

论文明确指出，在开启机密计算后，GPU内部的计算性能几乎没有损失。

- BF16矩阵乘法：在NVIDIA B300上，开启机密计算后的性能是未开启时的 99.8% (0.998x)。
- CUDA图：连续执行96,000次矩阵乘法的性能甚至达到了未开启时的 100.12% (1.0012x)。

这印证了加密和解密操作对Tensor Core等计算单元几乎没有影响。

所有的性能损失都发生在CPU和GPU之间的数据搬运（Host/Device Movement） 环节，论文将这一路径称为 “机密VM-GPU桥”（confidential VM-GPU bridge）。开启TEE后，这条“桥”出现了严重问题：

- 串行化（Serialized）：原本可以并行执行的多个数据搬运任务，在TEE下被强制串行化，失去了并发能力。多个CUDA流同时进行小数据传输时，并行度几乎为零。

- 异步失效（Async Revoked）：标准的异步拷贝指令（如 copy_(..., non_blocking=True)）在TEE下会退化为同步操作，阻塞CPU线程直到拷贝完成。

- 高昂的固定开销（Fixed Toll）：每一次数据搬运，都需要经过一个加密的中间缓冲区（staging buffer），产生约 330微秒 的固定设置成本。对于大量的小数据块传输，这个开销是致命的。

这些变化破坏了现代推理框架（如vLLM）依赖并行DMA来优化性能的假设。

这些“桥”上的问题最终导致了严重的端到端性能下降。摘要中提到的数据，正是这些具体问题的体现：

- 模型加载慢 34 倍 (34x model-load slowdown)：加载一个GPT-OSS-120B模型，在TEE下耗时 287秒，而理论上硬件可以在不到一分钟内完成。这是因为模型加载涉及海量的小块数据从CPU内存拷贝到GPU显存，每一次拷贝都要支付昂贵的固定开销。论文通过优化（如使用安全上下文池）成功将加载时间从287秒降至 8.4秒，证实了问题根源。

- KV缓存恢复延迟增加 131% (+131% KV-restore penalty)：同样是因为数据搬运路径的低效。

- LLM服务吞吐量下降 13-27%：以vLLM为例，其默认的异步调度策略在非机密环境下能提升性能，但在TEE下反而因“异步失效”而拖慢速度，导致 13-27% 的吞吐量损失。

开启TEE后，性能下降是真实且显著的，但它并非源于计算本身，而是源于CPU与GPU之间的数据搬运通道变得低效。

好消息是，论文指出这些问题是可以通过软件优化来缓解甚至解决的。例如，通过调整调度策略（--no-async-scheduling）可以恢复 57% 的性能差距；通过将阻塞操作移至独立线程，甚至可以恢复高达 92% 的性能差距。这说明，虽然TEE带来了新的性能挑战，但也存在相应的优化路径。

### 3.既然是NVLink 链路本身集成了专用加密/解密硬件，不需要把数据搬到别的慢路径处理,那把Hopper上的NVLink链路上换成这个Blackwell 上的NVLink链路是不是也能实现TEE-I/O功能？

Blackwell 的 NVLink encryption 不是一个可外挂的加密通道，而是集成在 NVLink endpoint 和 confidential computing stack 中的硬件能力。Hopper PPCIe 模式已经能保护 CPU-GPU PCIe 传输，但其 GPU-GPU NVLink/NVSwitch 通信不加密；Blackwell MPT CC 则在同一个 CVM 内支持 encrypted NVLink P2P。因此，不能简单通过“把 Hopper 上的 NVLink 换成 Blackwell NVLink”来实现 TEE-I/O，因为 TEE-I/O 还需要 endpoint crypto engine、secure firmware、key establishment、attestation 和驱动支持。Hopper 缺的是整套安全路径，而不是一根链路。



总结一下，Hopper PPCIe 主要解决 PCIe 上 CPU-GPU 通信保护，但不保护 NVLink；Blackwell MPT CC 才进一步支持多 GPU CVM 内的加密 NVLink P2P 通信。





---

## 四.超级服务器

NVIDIA GB300 NVL72 rack-scale system，这玩意里面有36 Grace Blackwell Superchips,提供1.1  exaFLOPS dense FP4 compute，很夸张。

---

## 五.卡的对比

RTX Pro 6000 Blackwell、RTX 5090 和 B300 虽然都基于NVIDIA的Blackwell架构，但它们的定位截然不同：**RTX Pro 6000是面向专业工作站的“生产力工具”，RTX 5090则是为发烧友和游戏玩家打造的“性能猛兽”，而B300则是面向数据中心AI工厂的“机架级AI加速器”**。

下面是它们的核心规格对比：

| 特性 | **NVIDIA RTX Pro 6000 Blackwell** | **NVIDIA GeForce RTX 5090** | **NVIDIA B300 / Blackwell Ultra** |
| :--- | :--- | :--- | :--- |
| **目标用户** | 专业设计师、AI研究员、工程师、科学家 | 游戏玩家、内容创作者、硬件发烧友 | 数据中心、AI工厂、大模型训练与推理 |
| **GPU 核心** | GB202 | GB202 | Blackwell Ultra，双reticle设计 |
| **CUDA 核心** | **24,064** 个 | 21,760 个 | 最多160个SM，640个第五代Tensor Core |
| **显存容量** | **96 GB** GDDR7 | 32 GB GDDR7 | **288 GB** HBM3E |
| **显存特性** | **支持 ECC (错误校正码)** | 不支持 ECC | HBM3E，支持MIG、机密计算和机架级NVLink互联 |
| **显存带宽** | 1,792 GB/s | 1,792 GB/s | **8 TB/s** |
| **单精度浮点 (FP32)** | **125 TFLOPS** | ~103 TFLOPS | 不是主要卖点，B300更强调Tensor Core低精度吞吐 |
| **AI 算力 (INT4/FP4)** | **4000 TOPS** | 未公布，约 2375 TOPS | **15 PFLOPS dense NVFP4**，20 PFLOPS sparse NVFP4 |
| **互联能力** | PCIe，工作站多卡生态 | PCIe，消费级平台 | NVLink 5，单GPU 1.8 TB/s双向，NVL72机架聚合130 TB/s |
| **功耗 (TDP)** | 600W | 575W | 最高 **1400W** TGP |
| **参考价格** | ~4,200 美元 | 1,999 美元 | 不单独面向消费市场销售，通常以HGX/DGX/GB300 NVL72系统交付 |

### 🚀 性能与算力：专业卡为何更强？

尽管RTX Pro 6000和RTX 5090核心相同，但RTX Pro 6000在规格上全面领先：

*   **更多核心**：它拥有 **24,064** 个CUDA核心，比RTX 5090的21,760个多出约10.6%。
*   **算力更强**：其单精度(FP32)算力达到 **125 TFLOPS**，高于RTX 5090的约103 TFLOPS。在AI推理方面，其INT4/FP4算力高达 **4000 TOPS**，远超RTX 5090。
*   **显存是杀手锏**：**96GB** 的显存是RTX 5090的**三倍**，能轻松运行 **700亿参数** 以上的大模型。并且支持 **ECC内存**，能自动纠正数据错误，保证长时间计算任务的稳定性。

如果把B300也放进来，它和前两张卡其实不是同一个市场：

*   **显存规模完全不同**：B300单卡有 **288GB HBM3E**，是RTX Pro 6000的3倍，也是RTX 5090的9倍。
*   **带宽完全不同**：B300的HBM3E带宽达到 **8 TB/s**，主要服务于大模型训练、长上下文推理和高并发推理。
*   **AI吞吐完全不同**：B300的重点不是FP32跑分，而是第五代Tensor Core、NVFP4、FP8、TMEM和NVLink 5带来的机架级吞吐。
*   **部署方式完全不同**：RTX Pro 6000和RTX 5090是单机/工作站卡，B300通常出现在HGX、DGX B300或GB300 NVL72这种数据中心系统里。

### 🛠️ 稳定性与生态：专业卡的护城河

RTX Pro 6000和B300的价值不仅在于硬件：

*   **ISV认证**：它通过了大量专业软件的 **ISV（独立软件开发商）认证**，确保在运行SolidWorks、Creo等工业软件时稳定高效，这是其高溢价的来源之一。
*   **企业级驱动**：使用专属的 **NVIDIA Enterprise 驱动**，提供更严格的稳定性测试和长期支持。
*   **虚拟化支持**：支持 **MIG（多实例GPU）** 技术，可将一张物理GPU虚拟为多个独立实例，提升资源利用率。
*   **数据中心级互联**：B300支持NVLink 5和NVLink Switch，目标不是单卡性能，而是让几十张GPU在一个机架内像统一的大加速器一样工作。

相比之下，RTX 5090虽然跑分不低，但在专业软件的稳定性和长期可靠性上无法与RTX Pro 6000相比；而B300则进一步跳出了“显卡”的范畴，更接近数据中心AI基础设施的一部分。





## 参考链接：

### 技术博客与资料

1. NVIDIA Blackwell Ultra 技术博客：https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
2. RTX Pro 6000 Blackwell 与 FlashAttention 4 相关资料：https://www.hardware-corner.net/rtx-pro-6000-blackwell-flashattention-4/

### NVIDIA Developer Forums

1. Hopper Protected PCIe mode 与 NVLink 安全讨论：https://forums.developer.nvidia.com/t/security-issues-in-protected-pcie-mode/323169
2. Blackwell NVLink encryption 讨论：https://forums.developer.nvidia.com/t/how-does-blackwell-support-high-performace-nvlink-encryption/337614

### NVIDIA 文档

1. Trusted Computing Solutions Release Notes：https://docs.nvidia.com/595trd1-trusted-computing-solutions-release-notes.pdf
2. Confidential Computing Deployment Guide with Intel TDX：https://docs.nvidia.com/cc-deployment-guide-tdx.pdf

### 论文

1. The Serialized Bridge: Understanding and Recovering LLM Serving Performance under Blackwell GPU Confidential Computing：https://arxiv.org/html/2606.23969v1
