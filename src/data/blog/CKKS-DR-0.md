---
title: "深度解析：CKKS 同态加密中的多精度乘法优化 (HMPM)"
pubDatetime: 2025-12-29T10:00:00Z
description: >
  本文基于论文《Homomorphic Multiple Precision Multiplication for CKKS and Reduced Modulus Consumption》，深入探讨如何通过同态欧几里得除法和密文分解技术，在 CKKS 方案中实现低模数消耗的高精度乘法。
tags:
  - FHE
  - CKKS
  - Cryptography
featured: true
draft: false
timezone: "Asia/Shanghai"
---

## 前言

全同态加密（FHE）允许我们在密文状态下进行计算，其中 CKKS 方案因其支持近似计算（Approximate Arithmetic）而特别适用于机器学习和隐私保护数据分析。然而，CKKS 在处理**高精度乘法**时面临一个严峻的挑战：**模数消耗（Modulus Consumption）**。

在标准 CKKS 中，为了维持高精度（例如 100-bit 的 Scaling Factor），每次乘法后的 Rescaling 操作需要消耗大量的模数位宽。这导致模数链迅速耗尽，限制了电路的深度，或者迫使我们使用巨大的环维数 $N$，从而拖慢计算速度。

本文将深入解析论文 **《Homomorphic Multiple Precision Multiplication for CKKS and Reduced Modulus Consumption》**（简称 HMPM 方案），探讨其如何通过**同态欧几里得除法（Homomorphic Euclidean Division）**和**双密文表示（Pair Representation）**，在保持高精度的同时，将模数消耗减半。

---

## 核心痛点：为什么高精度 CKKS 这么“贵”？

在 CKKS 方案中，消息 $m$ 被编码为 $m \cdot \Delta$（其中 $\Delta$ 是缩放因子）。
当我们执行乘法 $ct_1 \times ct_2$ 时，底层的消息变成了 $(m_1 \Delta) \times (m_2 \Delta) = m_1 m_2 \Delta^2$。
为了进行后续计算，我们必须通过 **Rescaling** 操作将缩放因子从 $\Delta^2$ 降回 $\Delta$。

*   **物理代价**：Rescaling 本质上是除法 $c \mapsto \lfloor c/q \rfloor$。在 RNS-CKKS 实现中，这意味着我们必须“牺牲”掉一层模数 $q$。
*   **精度与模数的关系**：如果你需要 100-bit 的精度（$\Delta = 2^{100}$），你就必须消耗掉 100-bit 的模数。
*   **后果**：为了支持深层的高精度计算，初始模数 $Q$ 必须极大，这直接导致环维数 $N$ 膨胀（例如从 $2^{15}$ 涨到 $2^{16}$ 或更高），计算延迟和存储开销成倍增加。

---

## 核心技术：同态欧几里得除法与 Pair 表示

HMPM 方案的核心思想借鉴了经典计算机科学中的**定点数算术（Fixed-point Arithmetic）**：当寄存器无法容纳大数乘法时，我们将大数拆分为“高位”和“低位”分别计算。

### 1. 密文分解 (Decomposition)

论文提出将一个大模数下的密文 $ct$，分解为两个较小模数空间下的分量：
$$ ct = 2^k \cdot \hat{ct} + \check{ct} $$

*   **$\hat{ct}$ (High Part / Quotient)**：商，代表数值的高位。
*   **$\check{ct}$ (Low Part / Remainder)**：余数，代表数值的低位。

这种分解被称为**同态欧几里得除法**。虽然在同态加密的噪声环境下，我们无法得到完美的数学商和余数，但论文证明了一种**“弱形式”（Weak Form）**的分解是可行的：即允许 $\hat{ct}$ 和 $\check{ct}$ 各自带有误差，只要它们组合起来能还原原始消息，且低位部分的系数足够小（$\le 2^k/2$）即可。

### 2. $\text{Mult}^2$ 算法：省模数的秘密

基于分解后的密文对 $(\hat{ct}, \check{ct})$，论文设计了全新的乘法算法 $\text{Mult}^2$。假设我们要计算 $A \times B$：
$$ A \approx 2^k \hat{A} + \check{A} $$
$$ B \approx 2^k \hat{B} + \check{B} $$

标准乘法展开为：
$$ A \cdot B = 2^{2k}(\hat{A}\hat{B}) + 2^k(\hat{A}\check{B} + \check{A}\hat{B}) + \check{A}\check{B} $$

**$\text{Mult}^2$ 的策略是：直接丢弃最后那一项 $\check{A}\check{B}$（低位 $\times$ 低位）。**

这样做有两个巨大的好处：
1.  **隐式除法（Implicit Division）**：通过提取高位和交叉项，我们实际上构造了一个在数学上被“除以了 $2^k$”的新结果。这相当于**免费**完成了一半的 Rescaling 工作。
2.  **模数节省**：由于 $\Delta \approx 2^{2k}$，我们只需要物理消耗剩下的 $2^k$ 大小的模数来进行后续的 $\text{RS}^2$（Rescaling）。

**结论**：原本需要消耗 $2k$ bits 模数的乘法，现在只需要消耗 $k$ bits。**模数消耗减半！**

---

## 算法详解：Tensor², Relin² 与 RS²

为了实现上述策略，算法被细分为三个步骤：

### 1. $\text{Tensor}^2$ (交叉乘法)
这一步负责计算 $\hat{A}\hat{B}$（新高位）和 $\hat{A}\check{B} + \check{A}\hat{B}$（新低位）。
*   **特点**：完全在 NTT 域（点值表示）进行，无需复杂的 NTT/INTT 变换，计算开销极低。
*   **误差引入**：丢弃的 $\check{A}\check{B}$ 项构成了主要的计算误差。只要 $2^k$ 足够大，这个误差就是可忽略的。

### 2. $\text{Relin}^2$ (重线性化)
标准的 Relinearization 会引入噪声。如果直接对高位 $\hat{ct}$ 做 Relin，噪声会被系数 $2^k$ 放大，导致精度崩溃。
*   **改进策略**：先将高位乘上 $2^k$（还原到原始量级），在整体上做 Relin，然后再分解。
*   **效果**：利用代数结构，将 Relin 产生的噪声“挤”到低位去，避免了高位噪声的放大。

### 3. $\text{RS}^2$ (重缩放)
这是物理消耗模数的一步。
*   **逻辑**：$\text{RS}^2(\hat{ct}, \check{ct})$ 利用“总数 - 高位 = 低位”的逻辑，强制让低位吸收掉 Rescaling 过程中的舍入误差。
*   **模数消耗**：仅消耗 $q_\ell$（约 $k$ bits），而不是完整的 $\Delta$（$2k$ bits）。

---

## 精度控制与参数优化

### 1. 为什么是 50/50 分解？
可能会有疑问：既然 $q_{\text{div}}$（分解因子）是“免费”的缩放，为什么不让它尽可能大？
*   **制衡**：$q_{\text{div}}$ 越大，被丢弃的误差项 $\frac{\check{A}\check{B}}{q_{\text{div}}}$ 中的分子（$\check{A}, \check{B} \propto q_{\text{div}}$）增长得比分母快。
*   **数学极限**：为了保证误差不爆炸，物理模数 $q_\ell$ 必须至少与 $q_{\text{div}}$ 一样大。
*   **最优解**：当 $q_{\text{div}} \approx q_\ell \approx \sqrt{\Delta}$ 时，模数消耗最小。这就是**50/50 对称分解**的由来。

### 2. 低位膨胀与 Recombine
随着乘法层数的增加，低位部分 $\check{ct}$ 会逐渐混入高位的信息（交叉项），导致其数值不断膨胀（每层增加约 1 bit）。
*   **风险**：如果低位太大，下次丢弃 $\check{A}\check{B}$ 时就会造成显著误差。
*   **解决方案**：**Recombine & Decompose (RCB $\circ$ DCP)**。定期（例如每 6 层）将高低位合并再重新分解，强制将低位“归零”回小的余数状态。

---

## 实验结果与应用价值

### 1. 更深的计算深度
在同等参数（$N=2^{15}$）下，标准 CKKS 只能支持 13 层乘法，而 Double-CKKS（即 $\text{Mult}^2$）可以支持 **18 层**。这是因为每层“省吃俭用”，同样的模数预算可以花得更久。

### 2. 高精度场景的降维打击
对于 100-bit 精度的需求：
*   **标准 CKKS**：需要消耗 100-bit 模数/层 $\rightarrow$ 总模数巨大 $\rightarrow$ 必须用 $N=2^{16}$ $\rightarrow$ 慢、大。
*   **HMPM 方案**：只需 50-bit 模数/层 $\rightarrow$ 总模数减小 $\rightarrow$ 可以用 **$N=2^{15}$** $\rightarrow$ **快、小**。
    *   **速度提升**：延迟降低约 1.5 倍。
    *   **存储节省**：密文大小仅为原来的 1/3。

### 3. 工程启示：32-bit RNS 系统
这篇论文不仅优化了高精度计算，还为工程实现打开了新思路。
通常 RNS-CKKS 依赖 50-60 bit 的大素数。结合 HMPM，我们可以使用 **28-bit 以下的小素数**（适配 32-bit 整数运算和 AVX2/512 指令集）来合成高精度 Scaling Factor。这允许在更广泛的硬件平台上实现高性能的 FHE。

---

## 总结

HMPM 方案通过巧妙的代数分解，打破了 CKKS 中“高精度 = 高消耗”的魔咒。它通过将乘法中的部分缩放任务转移给算法结构（丢弃低阶项），实现了模数消耗的减半。这不仅提升了同态计算的容量，更重要的是，它允许我们在更小的环维数下处理高精度任务，为全同态加密的实用化铺平了道路。

