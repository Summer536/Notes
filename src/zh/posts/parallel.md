---
title: 并行方法
date: 2025-06-25
readingTime: 300
category:
  - 笔记
tag:
  - GPU优化
  - 算法优化
# cover: /assets/images/cover3.jpg
isOriginal: true
---

# 并行方法 

## 简介

本文将讲解四大并行方法：
Data Parallel -> Model Parallel -> Pipeline Parallel -> Tensor Parallel

给你8张卡、16张卡DP怎么做？TP怎么做？EP（专家并行）怎么做？甚至还要了解一下SP（序列并行）。

如 tensor_parallel_size、pipeline_parallel_size、enable_expert_parallel、data_parallel_size）来手动设置 TP、PP、EP、DP 等并行策略

<!-- more -->

## 一、Data Parallel



## 二、Model Parallel


## 三、Pipeline Parallel


## 四、Tensor Parallel


## 附录：Ring-AllReduce

AllReduce 的最终目标，就是让每块 GPU 上的数据都变成下图箭头右边汇总的结果。

![](Figure/parallel/all_reduce.jpg)


Ring-AllReduce 是由百度提出的一种高效 **All Reduce** 算法，用于在分布式系统中进行数据同步。它通过环形拓扑结构，将数据分发到相邻的节点，从而实现高效的通信。
nvidia的NCCL通信库采用了这种算法。其通信流程如下图所示：

![](Figure/parallel/ringallreduce.jpg)

接下来计算的通信量只包括发送的参数量。假设有 $N$ 个设备，模型参数总量为 $\Psi$，每个梯度块的大小为 $\Psi/N$，每个设备只与其相邻的设备进行通信，首先讲解 Reduce-scatter 阶段：

- **步骤1**：显卡 a 将 a0 发送给显卡 b，同时接受显卡 d 发送的 d3。
- **步骤2**：显卡 a 将 a3 + d3 发送给显卡 b，同时接受显卡 d 发送的 c2 + d2。
- **步骤3**：显卡 a 将 a2 + c2 + d2 发送给显卡 b，同时接受显卡 d 发送的 b1 + c1 + d1。

Scatter-Reduce 阶段通信量：每次通信量是 $\Psi/N$，一共进行 $N-1$ 次通信，总通信量为 $\Psi*(N-1)/N$。

接下来介绍 AllGather 阶段：

- **步骤1**：显卡 a 将 a1 + b1 + c1 + d1 发送给显卡 b，显卡 b 直接做替换，同时接受显卡 d 发送的 a0 + b0 + c0 + d0，直接做替换。
- **步骤2**：显卡 a 将 a0 + b0 + c0 + d0 发送给显卡 b，显卡 b 直接做替换，同时接受显卡 d 发送的 a3 + b3 + c3 + d3，直接做替换。
- **步骤3**：显卡 a 将 a3 + b3 + c3 + d3 发送给显卡 b，显卡 b 直接做替换，同时接受显卡 d 发送的 a2 + b2 + c2 + d2，直接做替换。

AllGather 阶段通信量：同样的，每次通信量是 $\Psi/N$，一共进行 $N-1$ 次通信，总通信量为 $\Psi*(N-1)/N$。

可以看到，**单个设备通信量与 GPU 数量 $N$ 无关**，总通信量为：
$$
\Psi*(N-1)/N + \Psi*(N-1)/N = 2\Psi*(N-1)/N
$$

**当GPU数量足够多时，总通信量趋近于 $2\Psi$，即总通信量与 GPU 数量无关。**

值得注意的是，使用张量并行加速时，分布式系统 Allreduce 的**通信速度只受限于逻辑环中最慢的两个 GPU 的连接**;（每次需要通信的数据大小仅为 $\Psi/N$，随着 $N$ 增大，通信量减少，一般小于 network bandwidth）；总结就是 Ring Allreduce 的通信速度恒定，和设备数量无关，完全由系统中GPU 之间最慢的连接决定。

## 总结



## 待更新

## 参考资料


[大模型的分布式训练框架：deepspeed](https://mp.weixin.qq.com/s/kYeNjMsesfKfoZtJPRkciA)

[深度学习常见AllReduce算法图解](https://zhuanlan.zhihu.com/p/469942194)
