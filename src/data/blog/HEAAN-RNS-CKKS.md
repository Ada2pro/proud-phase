---
title: "阅读论文 A Full RNS Variant of Approximate  Homomorphic Encryption"
pubDatetime: 2026-01-03T10:00:00Z
description: >
  介绍2019年的论文《A Full RNS Variant of Approximate  Homomorphic Encryption》
tags:
  - FHE
  - CKKS
featured: true
draft: false
timezone: "Asia/Shanghai"
---

这篇论文 **《A Full RNS Variant of Approximate Homomorphic Encryption》**（Cheon 等，SAC 2018）解决了 CKKS/HEAAN 早期实现的一个关键工程瓶颈：**如何在不依赖大整数库的前提下，让 CKKS 的核心流程完全运行在 RNS（剩余类系统）+ NTT 的字长运算上**。

如果你只记住一句话：**它允许中间过程带着“$Q$ 的倍数噪声”跑，最终回到模 $Q$ 时这些噪声会自然消失；真正付出的代价主要是 ModDown（除法/取整）处的微小近似误差。**

## 目录

- [1. 背景：原版 CKKS 为什么难以 RNS 化？](#sec-1)
- [2. Full RNS 的核心改法：用“可控近似”换“字长性能”](#sec-2)
- [3. 关键子程序：快速（近似）基变换在算什么？](#sec-3)
- [4. 主戏：密钥切换里，为什么 $Q\cdot e$ 不会把结果毁掉？](#sec-4)
- [5. 真正的代价：误差从哪里来？](#sec-5)
- [Part 2：Rescale 与 ModDown 的本质区别](#part-2)
  - [1. 灵魂拷问：目的与对象的根本不同](#part-2-1)
  - [2. 算法实现的鸿沟：标量 vs. 向量](#part-2-2)
  - [3. 深度思考：为什么不能“连续 Rescale”？](#part-2-3)
  - [总结：一图胜千言](#part-2-summary)
- [Part 3：来自 WAHC 2023 的进阶思考（精度-效率权衡）](#part-3)
  - [1. 场景重构：当 Rescale 也变成了“多素数”问题](#part-3-1)
  - [2. 巅峰对决：逐步去除 vs. 一次性去除](#part-3-2)
  - [3. 结论与启示](#part-3-3)
  - [总结](#part-3-summary)
- [6. 性能结论（论文数据）](#sec-6)
- [7. 小结](#sec-7)

<a id="sec-1"></a>
## 1. 背景：原版 CKKS 为什么难以 RNS 化？

- **CKKS 的优势**：支持近似实数运算，对机器学习/统计计算很友好。
- **早期实现的痛点**：为了做 Rescaling（重缩放/取整），实现上常采用 $Q_L=q^L$ 这类模数链结构，导致模数是“大整数”，难以拆成互素小模数的乘积来做 RNS。
- **结果**：不得不依赖 GMP 等多精度大整数算术；而 BFV/BGV 等方案已经能用 RNS+NTT 把大头算子变成 64 位运算，CKKS 当时却吃不到这波性能红利。

<a id="sec-2"></a>
## 2. Full RNS 的核心改法：用“可控近似”换“字长性能”

论文提出全 RNS 变体（Full RNS Variant），核心是两点：

### 2.1 近似模数链（Approximate Basis）

不再强制 $Q$ 必须是某个缩放因子 $q$ 的严格幂，而是选取一组互素素数作为 RNS 基：

- $\mathcal{C}=\{q_0,\dots,q_{\ell-1}\}$，$Q=\prod_j q_j$
- 让每个 $q_j$ **近似等于**同一尺度（直觉上：都“差不多大”，便于层级管理与 rescale）

这会引入额外误差，但论文论证：只要参数设置得当（误差低于明文有效精度），结果仍可用。

### 2.2 近似模数切换（Approximate Modulus Switching）

为了让密钥切换（Key Switching）等过程能在 RNS 域完成，论文给出可在 RNS 上执行的近似：

- **ModUp**：从模 $Q$ 提升到模 $PQ$
- **ModDown**：从模 $PQ$ 降回模 $Q$，并伴随“除 $P$”的缩放

工程上通常还会选择满足 $q_i \equiv 1 \pmod{2N}$ 的素数，使每个 RNS 分量上都能高效做 NTT。

<a id="sec-3"></a>
## 3. 关键子程序：快速（近似）基变换在算什么？

在 RNS 中，一个整数/系数 $X$ 表示为：

$$
X \leftrightarrow (x_0,\dots,x_{\ell-1}),\quad x_i = X \bmod q_i
$$

Full RNS 的目标是避免“把 RNS 重构回大整数再处理”。它使用一种快速基变换（fast basis conversion）思路，直接在字长模运算里完成“从一个基到另一个基”的映射。

设两套 RNS 基：

- $\mathcal{C}=\{q_0,\dots,q_{\ell-1}\}$，模数积 $Q$
- $\mathcal{B}=\{p_0,\dots,p_{k-1}\}$，模数积 $P$

给定 $[a]_{\mathcal{C}}$ 计算 $[a]_{\mathcal{B}}$ 时，一种常见写法为：

$$
\mathrm{Conv}_{\mathcal{C}\to\mathcal{B}}([a]_{\mathcal{C}})
=
\left(
\sum_{j=0}^{\ell-1}
[a^{(j)}\cdot \widehat{q}_j^{-1}]_{q_j}\cdot \widehat{q}_j
\bmod p_i
\right)_{0\le i<k},
\ \widehat{q}_j = Q/q_j
$$

这一步“近似”的含义是：它对应的整数 $\tilde a$ 往往满足

$$
\tilde a = a + Q\cdot e
$$

其中 $e$ 很小。虽然 $Q\cdot e$ 看起来可能很大，但在模 $Q$ 下它是“隐形”的：

$$
a + Q\cdot e \equiv a \pmod Q
$$

这就是后面“噪声消失”的关键伏笔。

<a id="sec-4"></a>
## 4. 主戏：密钥切换里，为什么 $Q\cdot e$ 不会把结果毁掉？

密钥切换/重线性化通常需要先把模从 $Q$ 扩到 $PQ$，运算后再降回 $Q$。

### 4.1 近似 ModUp：从 $Q$ 提升到 $PQ$

对 $c\in R_Q$ 做近似提升得到 $\tilde c \in R_{PQ}$：

$$
\tilde c = c + Q\cdot v
$$

其中 $v$ 很小，但 $Q\cdot v$ 形式上可能“看起来很吓人”。

### 4.2 与切换密钥相乘：噪声被“绑上”了 $P$

切换密钥可抽象为（省略若干细节项）：

$$
\mathrm{swk} \approx (-a's_2 + P\cdot s_1 + E,\ a') \pmod{PQ}
$$

内积里会出现：

$$
\mathrm{Term} = \tilde c\cdot (P\cdot s_1)\pmod{PQ}
$$

代入 $\tilde c=c+Q\cdot v$：

$$
\begin{aligned}
\mathrm{Term}
&= (c+Q\cdot v)\cdot (P\cdot s_1) \\
&= \underbrace{c\cdot P\cdot s_1}_{\text{信号项}}
\;+\;
\underbrace{Q\cdot v\cdot P\cdot s_1}_{\text{干扰项}}
\end{aligned}
$$

干扰项同时带有因子 $Q$ 与 $P$。

### 4.3 近似 ModDown：除以 $P$，并回到模 $Q$

最后要把结果降回 $R_Q$ 并“除以 $P$”：

- **信号项**：$\frac{c\cdot P\cdot s_1}{P}=c\cdot s_1$，正是目标。
- **干扰项**：$\frac{Q\cdot v\cdot P\cdot s_1}{P}=Q\cdot v\cdot s_1$，而

$$
Q\cdot v\cdot s_1 \equiv 0 \pmod Q
$$

因此，ModUp 阶段引入的“$Q$ 的倍数噪声”会在回到模 $Q$ 时自然归零。

<a id="sec-5"></a>
## 5. 真正的代价：误差从哪里来？

上面消掉的是 **$Q$ 的倍数噪声**。真正的近似误差主要来自 **ModDown 的除法/取整**：在 RNS 域里无法像大整数那样做完全精确的除 $P$，会出现

$$
\mathrm{Result} = \left\lfloor X/P \right\rfloor + e_{\mathrm{small}}
$$

其中 $e_{\mathrm{small}}$ 是很小的舍入误差。对 CKKS 来说，只要参数使该误差小于明文有效精度，这个代价通常是可接受的。

<a id="part-2"></a>
## 【硬核科普】全 RNS 同态加密(Part 2)：Rescale 与 ModDown 的本质区别——为何看似相同的除法，实现却天差地别？

在阅读全 RNS 变体（Full RNS Variant of HEAAN/CKKS）的论文或代码时，你可能会发现一个有趣的现象：**Rescale（重缩放）** 和 **ModDown（模数约减）** 在数学形式上惊人地相似。

它们本质上似乎都在做同一件事：

$$
c_{new} = \frac{c_{old} - (c_{old} \pmod M)}{M}
$$

即：减去余数，然后除以模数。

然而，在工程实现中，Rescale 只是几行简单的减法和乘法，而 ModDown 却需要调用复杂的“近似基变换”算法。为什么会有这种差异？能不能用连续执行 Rescale 的方式来替代 ModDown？

本文将带你从数学原理和工程优化两个维度，彻底理清这两个操作的爱恨纠葛。

<a id="part-2-1"></a>
### 1. 灵魂拷问：目的与对象的根本不同

虽然动作都是“除法”，但两者的动机和对消息（Message）的影响截然不同。

#### Rescale (重缩放)：为了“丢弃”

- **触发场景**：同态乘法之后。
- **除数**：密文模数链末端的素数 $q_\ell$（约 60 bit）。
- **动机**：在 CKKS 中，密文 $c$ 加密的消息是 $m = \Delta \cdot x$（$\Delta$ 是缩放因子）。两个密文相乘后，消息变成了 $\Delta^2 \cdot x^2$。如果不处理，缩放因子会指数级爆炸。
- **对消息的影响**：**变小了**。

$$
m_{new} \approx m_{old} / q_\ell \approx (\Delta^2 \cdot x^2) / \Delta = \Delta \cdot x^2
$$

我们是**故意**要除去一部分数据的（即那个多出来的 $\Delta$），以维持缩放比例恒定。

- **代价**：模数链层级永久减少一层（$L \to L-1$）。

#### ModDown (近似模数约减)：为了“还原”

- **触发场景**：密钥切换（KeySwitching）的最后一步。
- **除数**：辅助模数积 $P = p_0 \cdots p_{k-1}$（非常大的整数）。
- **动机**：为了在密钥切换中掩盖噪声，我们将密文提升到了 $PQ$ 空间，并且乘以了 $P$。此时消息的状态是 $P \cdot m$。现在我们需要把这个为了计算而临时引入的 $P$ 去掉。
- **对消息的影响**：**不变**。

$$
m_{new} \approx m_{old} / P = (P \cdot m) / P = m
$$

我们**不想**丢失任何消息信息，只是想把“保护壳”拆掉。

- **代价**：模数空间从扩展态 $PQ$ 回归到正常态 $Q$（不消耗层级）。

<a id="part-2-2"></a>
### 2. 算法实现的鸿沟：标量 vs. 向量

为什么 Rescale 很简单，而 ModDown 很复杂？核心在于**余数（Remainder）的大小**以及**我们在哪个基（Basis）上计算**。

#### Rescale：简单的标量广播

当我们做 Rescale 时，我们要除以 $q_\ell$。

- 余数 $r = c \pmod{q_\ell}$。
- 因为我们在 RNS 系统中，直接读取第 $\ell$ 个分量就得到了这个 $r$。
- $r$ 是一个 **60 bit 的小整数**。
- **操作**：直接把这个小整数 $r$ 拿去减其他分量即可。

$$
c_i \leftarrow (c_i - r) \cdot q_\ell^{-1} \pmod{q_i}
$$

#### ModDown：跨越维度的向量翻译

当我们做 ModDown 时，我们要除以 $P$。

- 余数 $R = c \pmod P$。
- 我们在 RNS 系统中，虽然拥有 $R$ 在基 $\mathcal{B}=\{p_0, \dots, p_k\}$ 下的表示（即密文的 $P$ 部分分量），但这对于基 $\mathcal{C}=\{q_0, \dots, q_\ell\}$ 来说是**不可读的**。
- **困境**：我们需要在 $Q$ 的地盘上减去 $R$，但我们手里只有 $R$ 在 $P$ 地盘上的“护照”。
- **解决方案**：必须使用 **基变换（Basis Conversion）**。

我们需要先把 $R$ 从基 $\mathcal{B}$ 翻译（转换）到基 $\mathcal{C}$，计算出 $R \pmod{q_i}$ 的值。这就需要用到论文中提到的 `Conv_B->C` 算法。

$$
\text{Target} = (c_{in\_Q} - \text{Convert}_{\mathcal{B} \to \mathcal{C}}(c_{in\_P})) \cdot P^{-1}
$$

这就是 ModDown 如此复杂的根本原因：**它涉及两个不同 RNS 基之间的数据搬运。**

<a id="part-2-3"></a>
### 3. 深度思考：为什么不能“连续 Rescale”？

既然 $P = p_0 \cdot p_1 \cdots$，我们能不能不像 Algorithm 2 那样一次性除以 $P$，而是像剥洋葱一样，先 Rescale $p_0$，再 Rescale $p_1$……直到把 $P$ 里的素数都除完？

**答案是：理论上可行，但工程上是“血亏”的。**

这种方法称为 **Sequential Rescaling（顺序重缩放）**，它比 **ModDown（批量处理）** 差在以下三点：

#### A. 算力的无谓浪费 (Lazy Reduction)

如果你决定先除以 $p_0$：

1. 你不仅要更新 $Q$ 部分的分量。
2. **你还必须更新 $p_1, p_2, \dots$ 这些 $P$ 部分的分量！**

为什么？因为为了下一步能继续除以 $p_1$，你必须知道当前密文在 $p_1$ 下的余数。而一旦你改变了密文（除以了 $p_0$），原来的余数就失效了，必须重新计算。

然而，这些 $P$ 分量的唯一宿命就是被丢弃。花费 $O(k^2)$ 的算力去维护这些“即将死亡”的模数分量，是极大的浪费。

**ModDown 的做法**：它在 $t=0$ 时刻一次性读取所有 $P$ 分量，算完直接把整个 $P$ 部分扔掉。它**从不更新** $P$ 部分的数据，实现了极致的“懒惰计算”。

#### B. 噪声结构的破坏

全 RNS 变体依赖于 $a \equiv a + Q \cdot e \pmod Q$ 这一特性。

- **ModDown**：一次性除法引入的误差结构单一，易于分析证明其在模 $Q$ 下会消失。
- **连续 Rescale**：每一步都会引入舍入误差，且误差会被后续的除法反复缩放。这会导致误差分析变得极其复杂，且可能破坏近似基变换所需的特定误差结构。

<a id="part-2-summary"></a>
### 总结：一图胜千言

| 特性 | Rescale (重缩放) | ModDown (Algo 2) |
| :--- | :--- | :--- |
| **形象比喻** | **科学计数法调整**<br>($1.23 \times 10^2 \to 12.3 \times 10^1$) | **拆快递**<br>(把为了运输加上的保护箱 P 拆掉) |
| **数学除数** | 单个小素数 $q_\ell$ | 模数积 $P$ (一组素数的乘积) |
| **余数获取** | 直接读取 ($O(1)$) | 需要基变换 ($O(k\ell)$) |
| **对模数链** | 降级 (不可逆) | 恢复 (回到计算层) |
| **工程策略** | 逐层剥离 | 批量处理 (Lazy) |

看懂了 Rescale 和 ModDown 的区别，你就真正看懂了 CKKS 全 RNS 优化的核心逻辑。前者是算法正确性的保证（控制 Scale），后者是工程性能的保证（快速 KeySwitch）。

*(未完待续...)*

<a id="part-3"></a>
## 【进阶思考】批量处理总是最好的吗？——来自 WAHC 2023 的反直觉启示

这里顺便澄清一下你可能在笔记/代码里看到的缩写：**WAHC 指代的论文是**：

- **"High-precision RNS-CKKS on fixed but smaller word-size architectures: theory and application"（WAHC ’23）**

从**数学结构**上看，WAHC 2023 论文中的 **Composite Scaling（$t$ 个小素数组成一个 $q_\ell$）** 与常规 CKKS 中的 **KeySwitching 辅助模数（$k$ 个小素数组成 $P$）** 是高度同构的：

它们面临的本质问题都是：**如何在一个 RNS 系统中，除以一个由多个小素数组成的“大整数”？**

- **常规 ModDown**：$M=\prod_{i=0}^{k-1} p_i$（为了去掉 $P$）。
- **Composite Rescale**：$M=\prod_{j=0}^{t-1} q_{\ell,j}$（为了去掉 $\Delta$）。

因此，论文关于“逐步处理（Iterative）误差小于批量处理（Batch FBC）”的结论，理论上也可以迁移到 ModDown（去 $P$）的讨论中——这是一个很棒的**进阶/延伸**知识点。

在上一节中，我们对比了 **Rescale（单素数除法）** 和 **ModDown（多素数批量除法）**，并指出 ModDown 采用了“一次性打包（Batch）”的策略来追求效率。

然而，**“批量处理”在任何情况下都是最优解吗？**

如果我们的硬件受到限制（例如只有 32 位字长），或者我们对精度的要求到了极致，情况会发生什么变化？

这正是 WAHC ’23 想回答的问题：它揭示了 RNS 除法中一个鲜为人知的 **精度-效率权衡（Precision-Efficiency Trade-off）**。

<a id="part-3-1"></a>
### 1. 场景重构：当 Rescale 也变成了“多素数”问题

在 32 位嵌入式设备或 GPU 上，单一的 32 位素数不足以提供足够的精度（通常 Scaling Factor $\Delta \approx 2^{40} \sim 2^{60}$）。

为了解决这个问题，作者提出 **Composite Scaling（组合缩放）**：用 $t$ 个小的 32 位素数乘积来模拟一个大的 Scaling Factor。

$$
q_\ell = q_{\ell,0} \cdot q_{\ell,1} \cdots q_{\ell,t-1} \approx \Delta
$$

这就导致了一个有趣的现象：**现在的 Rescale 操作，数学上变得和 ModDown 一模一样了！** 它们都需要除以一个由多个 RNS 分量组成的积。

<a id="part-3-2"></a>
### 2. 巅峰对决：逐步去除 vs. 一次性去除

面对这个“多素数除法”问题，作者对比了两种策略：

- **策略 A（批量法 / Big FBC）**：

  类似于标准的 ModDown。利用大型快速基变换，把 $t$ 个素数看作整体，一次性算完。

  - *优点*：流程统一，理论计算量可能更低（取决于 $t$ 的大小）。
  - *缺点*：论文指出，其误差上界为 $\sqrt{t}\cdot B_{rs}$。

- **策略 B（逐步法 / Iterative Rescale）**：

  像剥洋葱一样，先 Rescale 掉 $q_{\ell,0}$，更新密文；再 Rescale 掉 $q_{\ell,1}$... 循环 $t$ 次。

  - *优点*：论文证明，其误差上界仅为 $B_{rs}$（与单次操作相当）。
  - *缺点*：需要维护中间状态，逻辑较繁琐。

<a id="part-3-3"></a>
### 3. 结论与启示

**惊人的结论：** 在 WAHC ’23 中，作者指出**“逐步法”在理论误差控制上优于“批量法”**（误差界小了 $\sqrt{t}$ 倍）。

这给我们带来了什么启示？

1. **对于 KeySwitching（ModDown）**：

   目前的标准库（SEAL、OpenFHE）依然使用**批量法**去除 $P$。

   - *原因*：$P$ 包含的素数数量 $k$ 通常不多（2～4 个），$\sqrt{k}$ 的误差放大往往可以接受。并且 KeySwitching 本身就是性能瓶颈，工程上更愿意用一点噪声换速度。

2. **对于受限硬件（Composite Scaling）**：

   当 $t$ 变大（例如为了凑 60 bit 精度，32 位机器需要更多小素数），或者在深度神经网络等对噪音极度敏感的任务中，**“逐步法”** 可能成为更好的选择。

<a id="part-3-summary"></a>
### 总结

- **ModDown 的现状**：为了快，我们选择一次性除掉 $P$（Batch ModDown），哪怕噪音稍微大一点点（$\sqrt{k}$ 倍）。
- **未来的可能性**：如果你在设计高精度、低字长的同态加速器，或者发现 KeySwitching 噪音成为了瓶颈，不妨回头试试“逐步 ModDown”——这可能是一条通过牺牲算力来换取精度的有效路径。

*此补充内容受 WAHC ’23 论文启发，特此致谢。*

<a id="sec-6"></a>
## 6. 性能结论（论文数据）

论文将 HEAAN-RNS 与早期 HEAAN 实现对比（Intel Core i5 单核）：

- **基础操作加速**
  - **解密**：约 **17.3×**（135ms → 7.8ms）
  - **常数乘法**：约 **6.4×**
  - **同态乘法**：约 **8.3×**（1355ms → 164ms）
- **应用场景**
  - 解析函数（逆/exp/sigmoid）：约 160ms（摊销后每 slot 约 20µs）
  - 统计函数：$2^{13}$ 个实数的均值/方差约 307ms / 518ms
  - 逻辑回归训练：575 样本（每样本 8 特征）单核约 1.8 分钟

这条路线后来也成为现代同态加密库实现 CKKS 的常见做法（如 Microsoft SEAL、OpenFHE、Lattigo 等）。

<a id="sec-7"></a>
## 7. 小结

全 RNS 变体的工程思想可以概括为：

1. 不再执着于中间步骤的“精确大整数重构”
2. 允许中间结果携带 $Q$ 的倍数误差，并利用同余结构在回到模 $Q$ 时自然消失
3. 把关键算子落到 RNS 分量的 64 位运算 + NTT 上，从而获得数量级加速

---

*本文基于 SAC 2018 论文 "A Full RNS Variant of Approximate Homomorphic Encryption" 整理。*

