---
title: Softmax
date: 2025-05-20
readingTime: 300
category:
  - 笔记
tag:
  - GPU优化
  - 算法优化
cover: /assets/images/cover3.jpg
isOriginal: true
---

# Softmax 

## 简介

本文将讲解Softmax发展过程：
naive softmax -> safe softmax -> online softmax

<!-- more -->

## Naive softmax

原始softmax的公式为:
$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

- 对于向量中的每一个元素，**它的MAC(memory access count)为3**: 第一次pass中load一次，在第二次pass中load一次, store一次, 所以一共是三次memory access。

- Original softmax的问题所在: 在第三行的算法中，对进行sum的过程中，由于真实硬件的浮点格式所能表示的范围限制(fp16正数所能表示的最大值为65504，而$e^{12}$>65504),很容易造成上溢或者下溢。
为了解决上述问题，从而引出safe softmax。

    ![Naive softmax](Figure/softmax/naive.png "Naive softmax")


## Safe softmax
为解决上述提到的可能的数据溢出问题，基本上所有的深度学习框架使用的都是safe Softmax的计算。其计算公式如下：
$$
c = max(z_1,z_2,...,z_n)
$$
$$
\sigma(z_i) = \frac{e^{z_i-c}}{\sum_{j=1}^{n} e^{z_j-c}}
$$
- 该计算公式在数学上和naive等价:
$$
\sigma(z_i) = \frac{e^{z_i-c}}{\sum_{j=1}^{n} e^{z_j-c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_{j=1}^{n} e^{z_j} \cdot e^{-c}} =  \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$
- Safe Softmax所带来的问题: 为了安全，我们需要额外求出输入向量中的元素最大值，这带来了多一次的循环pass，并且对于向量中的每一个元素，**它的MAC(memory access count)为4**。具体表现为在第一次pass中Load $ z_i $ 一次, 在第二次pass中Load $ z_j $ 一次，在第三次pass中Load $ z_i $ 一次, Store $ \sigma_{z_i} $ 一次, 所以总共mac是4次。
为了解决上述问题，从而引出了Online Softmax。

    ![Safe softmax](Figure/softmax/safe.png "Safe softmax")


## Online softmax

在Safe的基础上，Online softmax做出的主要改进为: 将最大值 $c = max(z_1,z_2,...,z_n)$ 和归一因子 $d = \sum_{j=1}^{n} e^{z_j-c}$ 放在同一个循环pass中处理。

### 具体实现

循环pass处理 $z_1$ -> $z_n$：
- if $z_i \leqslant c$ : 
    $$
    c_{new} = c_{old}
    $$
    $$
    d_{i} = d_{i-1} + e^{z_i-c_{new}}
    $$

- if $z_i > c$ :
    $$
    c_{new} = z_i
    $$
    之前计算的d是相较于旧的c即 $c_{old}$ 的，需要将其转换由新的 $c_{new}$计算:
    $$
    d_{old} = \sum_{j=1}^{n} e^{z_j-c_{old}}, \quad d_{new} = \sum_{j=1}^{n} e^{z_j-c_{new}}
    $$
    通过以下公式可进行转换:
    $$
    d_{new} = d_{old} \cdot \frac{e^{c_{old}}}{e^{c_{new}}}
    $$
    因此对于归一化因子d更新时:
    $$
    d_{i} = d_{new} + e^{z_j-c_{new}} = d_{new} + e^{z_j-z_j} = d_{new} + 1
    $$ 

<br><br>
该算法在迭代输入数组的元素时保留最大值c 和归一化项 d。在每次迭代中，它都需要将 normalizer d 调整为新的最大 cj，然后才向 normalizer 添加新的值。
**这里我们把vector中的每个元素的MAC从4降到了3**，在第一次pass里面，我们load一次 $ z_j $ 即可，在第二次pass里面我们load一次 $ z_i $ ,store一次 $ \sigma_{z_i} $,所以一共是3次memory access。

![Online softmax](Figure/softmax/online.png "online softmax")


## 参考文献

1. [Milakov M, Gimelshein N. Online normalizer calculation for softmax[J]. arXiv preprint arXiv:1805.02867, 2018](https://arxiv.org/pdf/1805.02867)

2. [从 Naive Softmax到Online Softmax and Top-k](https://zhuanlan.zhihu.com/p/1892986988065453222)