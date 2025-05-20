---
title: Flashattention
date: 2025-05-10
readingTime: 600
category:
  - 机器学习
  - 深度学习
  - 笔记
tag:
  - 注意力机制
  - GPU优化
  - 大模型
  - 算法优化
# cover: /assets/images/cover3.jpg
isOriginal: true
---

# Flashattention

## 简介

Flashattention是一种高效的注意力机制计算方法，由斯坦福大学研究团队在2022年提出。它通过优化GPU内存访问模式，显著提高了Transformer模型中注意力计算的速度并降低了内存使用。作为大模型训练和推理的关键优化技术，Flashattention已被广泛应用于各种大型语言模型中。

<!-- more -->

## 一、Flashattention-V1

### 1.1 标准注意力机制
给定输入二维矩阵 $Q, K, V \in \mathbb{R}^{N \times d}$，其中 $N$ 是输入序列的长度，$d$ 是自注意力机制头的长度。Softmax 是按行应用的，注意力输出矩阵 $O \in \mathbb{R}^{N \times d}$ 的计算公式如下：
$$
\begin{align*}
S &= Q K^\mathrm{T} \in \mathbb{R}^{N \times N}, \quad
P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad
O = P V \in \mathbb{R}^{N \times d}.
\end{align*}
$$
![](Figure/flashattention/FAV1_0.png)
标准的 Attention 运算大致可以描述为以下三个步骤：
- 将 Q,K 矩阵以块的形式从 HBM 中加载到 SRAM 中，计算 S=QK ，将 S 写入到 HBM 中。
- 将 S 矩阵从 HBM 中加载到 SRAM 中，计算 P=Softmax(S) ，将 P 写入到 HBM 中。
- 将 P,V 矩阵以块的形式从 HBM 中加载到 SRAM 中，计算 O=PV ，将 O 写入到 HBM 中。

![](Figure/flashattention/FAV1_2.png)

self-attention 算子涉及到的和 HBM 数据传输过程如上图所示，很明显需要从 HBM 中读取 5 次，写入 HBM 3 次，HBM 访存量 $MAC = 4N^2 + 3Nd$，很明显标准注意力的 HBM 访问代价 $MAC$ 随序列长度增加呈二次方增长。

而 self-attention 的计算量为 $4N^2d+N^2$，标准注意力算子的操作强度 $= \frac{4N^2d+N^2}{4N^2 + 3Nd}$。公式可看出，标准注意力算子是一个很明显的内存受限型算子。
![](Figure/flashattention/MACandFlops.png)

### 1.2 FAV1整体介绍

#### 挑战
在Flash Attention出来之前，已经有了很多了fusedattention算子，但是仔细看，可以发现这其实不是真正的融合算子，只是把matmul kernel、scale kernel、softmax kernel的接口在一个fusedattention算子里面按照计算顺序调了一下，这种手法最多减少了pytorch、TF等框架对算子的调度开销，其实不能真正解决对HBM或者显存的memory traffic。

**融合MHA的挑战在于两点：**

1.解决softmax，因为softmax是一个row-wise（以行为单位）的操作，必须要遍历softmax一行才能得到结果，由此，后面的matmul不得不等待这个过程，导致并行性降低

2.在寄存器和shared memory复用数据做计算，而不是去HBM或显存上去读数据来计算，然而寄存器数量和shared memory (也就是图中的SRAM) 大小都有限，在左图的情况下，显然无法将softmax的结果存到这两个存储单元里面供下一个matmul复用，下一个matmul不得不去HBM或显存上读数据

![](Figure/flashattention/FAV1_6.png)

#### 整体思想
FAV1算法整体做到了如下的两部分：
- Tiling（在前向和后向传递中使用）:
  1. （解决挑战1）Online Softmax 实现在一个 for 循环中计算 
$m_i$ 和 $d_i$ ，**FlashAttention-v1 基于它的思想更进一步，实现在一个 for 循环中计算 $m_i$ 、$d_i$ 和注意力输出 $O_i$ ，也就是说，在一个 kernel 中实现 attention 的所有操作**。

  2. （解决挑战2）再**通过分块 Tiling 技术**，将输入的 Q、K、V 矩阵拆分为多个块，将其从较慢的 HBM 加载到更快的 SRAM 中，从而大大减少了 HBM 访问次数（内存读/写的次数），然后分别计算这些块的注意力输出，最后，将每个块的输出按正确的归一化因子缩放之后相加后可得到精确的注意力输出。
- 重新计算（仅在后向传递中使用）:核心思想为不为反向传递存储S、P矩阵，但是输出softmax的l和m，在反向传播时，重新计算S、P矩阵。

### 1.3 FAV1算法流程

<!-- ![](Figure/flashattention/FAV1_1.png "FlashAttention Block Diagram") -->
<!-- <p align="center">
  <img src="Figure/FA1.png" width="500" alt="核心思想"/>
</p> -->
Flashattention总体的算法流程图如下：
![](Figure/flashattention/FAV1_5.png)


1. ![](Figure/flashattention/step1(1).png)
设置块的行大小 $B_r = \frac{M}{4d}$，块的列大小为 $B_c = \min\left(\frac{M}{4d}, d\right)$ 。 $\min$ 函数的目的是防止块大小 $B_r \times B_c > M/4$，这样就无法把 4 个这样的块放到 SRAM 里，后面我们会看到为什么是 $4 \times B_r \times B_c$ 的块。

2. ![](Figure/flashattention/step2.png)
我们把结果矩阵O初始化为零，后面会逐步把中间结果累加进去，所以零是合适的初始值。类似的是l(注意：对于每一行来说，它是一个标量，用于累加指数和，由于输出有N行，所以这里的l是长度为N的向量)。m用于记录每一行当前最大的值，所以也是长度为N，而-inf是求max的合适初始值。



![](Figure/flashattention/FAV1_9.png)
### 1.4 FAV1算法的数学证明
#### 1.4.1 Online Softmax
这个证明略，详细内容见笔记[Naive -> Safe -> Online Softmax](https://summer536.github.io/Notes/zh/posts/softmax.html)

#### 1.4.2 算法流程

#### 1.4.3 算法实现



## 二、Flashattention-V2

## 三、Flashattention-V3

## 总结

## 待更新

## 参考文献

1. [Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)
2. [Dao, T., et al. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ](https://arxiv.org/pdf/2307.08691)
3. [Flash Attention V1论文解读-李理](https://fancyerii.github.io/2023/10/23/flashattention/)
4. [Flash Attention V1论文解读-Zhang](https://www.armcvai.cn/2024-10-02/flashattention1-paper.html)
5. [Flash Attention V2论文解读-Zhang](https://www.armcvai.cn/2024-10-05/flashattention2-paper.html)
6. [Flash Attention V3论文解读-Zhang](https://www.armcvai.cn/2024-10-06/flashattention3-paper.html)
7. [flashattention1-2-3 系列总结-Zhang](https://www.armcvai.cn/2024-10-07/flashattention1-2-3-summary.html)
8. [Flash Attention1-真正意义上的scale dot product attention的算子融合-AI不止算法](https://mp.weixin.qq.com/s?__biz=Mzg2ODk4MzE2MQ==&mid=2247483875&idx=1&sn=a23ef737b03e5bdec1892a8818de0704&chksm=cea549f5f9d2c0e3832f10031de98dd4a243fd2411ffdacb434cc64a473452a148fc149ded47&scene=21#wechat_redirect)
9. [FlashAttentionV1V2算法解释-AI不止算法Bilibili](https://www.bilibili.com/video/BV1gzBqY4Evw?spm_id_from=333.788.videopod.sections&vd_source=f058beebb64c488b55915da416ee6086)