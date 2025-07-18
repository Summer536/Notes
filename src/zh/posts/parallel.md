---
title: 并行方法DP、TP、PP、EP、SP
date: 2025-07-15
readingTime: 300
category:
  - 笔记
tag:
  - GPU优化
  - 算法优化
# cover: /assets/images/cover3.jpg
isOriginal: true
---

# 并行方法DP、TP、PP、EP、SP

## 简介

给你8张卡、16张卡DP怎么做？TP怎么做？EP（专家并行）怎么做？甚至还要了解一下SP（序列并行）。

如 tensor_parallel_size、pipeline_parallel_size、enable_expert_parallel、data_parallel_size）来手动设置 TP、PP、EP、DP 等并行策略

<!-- more -->

## 一、Data Parallel
数据并行（Data Parallel）是数据平均分配到GPU上并行处理的策略。**每个GPU上都拥有一份完整的模型副本**，**各自吃一份数据(mini-batch)，算一份梯度，最后对梯度进行累加**来更新整体模型。

![](Figure/parallel/dp.png)

数据并行主要分为以下三种方法：
- DP（Data Parallelism）：最早的数据并行模式，一般采用参数服务器(Parameters Server)这一编程框架。实际中多用于**单机多卡**
- DDP（Distributed Data Parallelism）：分布式数据并行，采用Ring AllReduce的通讯方式，实际中多用于**多机多卡场景**
- ZeRO：零冗余优化器。由微软推出并应用于其DeepSpeed框架中。严格来讲ZeRO采用**数据并行+张量并行**的方式，旨在**降低存储**。

### 1.数据并行（DP）

#### 1.1 主要框架
![](Figure/parallel/DP.jpg)
1. 默认的主卡（通常是 GPU 0）负责读取一个 batch 的数据，并将数据划分为多个 mini-batch 分别发送到其他 GPU；
2. 从主卡（GPU 0）复制一份最新的模型到所有 GPU 上；
3. 每张 GPU 独立执行前向传播（FWD），得到各自的输出；所有 GPU 的输出被发送回主卡（GPU 0）进行 loss 计算；
4. 这个 loss 被广播到所有 GPU 进行反向传播（BWD）；
5. 每张 GPU 分别计算自己的梯度，并将这些梯度返回给 GPU 0；GPU 0 聚合计算所有梯度用于更新模型参数，并广播到所有 GPU 上。这个操作为**AllReduce**。


前文说过，实现DP的一种经典编程框架叫“参数服务器”，在这个框架里，**计算GPU称为Worker，梯度聚合GPU称为Server**。在实际应用中，为了尽量减少通讯量，一般可选择一个Worker同时作为Server。比如GPU0既做计算（Worker），也做梯度聚合（Server），如上图。需要再额外说明几点：

- 1个Worker或者Server下可以不止1块GPU。
- Server可以只做梯度聚合，也可以梯度聚合+全量参数更新一起做。

#### 1.2 优缺点
优点：
  - 实现简单，易于理解
  - 适合单机多卡（小模型）

两个主要问题：
  - 存储开销大。每块GPU上都存了一份完整的模型，造成冗余。**（ZeRO对其进行了优化）**
  - 通讯开销大。Server需要和每一个Worker进行梯度传输。当Server和Worker不在一台机器上时，Server的带宽将会成为整个系统的计算效率瓶颈。**（DDP对其进行了优化）**

#### 1.3 梯度异步更新
了解上述的DP计算过程，我们不难发现一个问题：当Server在搬运数据，计算梯度的时候，Worker们在干嘛呢？

当然是在：摸鱼！！！

为了尽可能的压榨计算资源，老板们想出了一个办法：**梯度异步更新**。

梯度异步更新简单来讲就是：在第N轮计算中，Worker正常计算梯度，并向Server发送梯度请求。但是，该Worker并不会实际等到把聚合梯度拿回来，更新完参数W后再做计算。而是直接拿旧的W，吃新的数据，继续第N+1轮的计算。这样就保证在通讯的时间里，Worker也在马不停蹄做计算，提升计算通讯比。
![](Figure/parallel/yibu.jpg)

当然，异步也不能太过份。只计算梯度，不更新权重，那模型就无法收敛。图中刻画的是延迟为1的异步更新，也就是在开始第12轮对的计算时，必须保证W已经用第10、11轮的梯度做完2次更新了。

参数服务器的框架下，**延迟的步数也可以由用户自己决定**，下图分别刻划了几种延迟情况：

![](Figure/parallel/yibu2.jpg)

- (a) 无延迟
- (b) 延迟但不指定延迟步数。也即在迭代2时，用的可能是老权重，也可能是新权重，听天由命。
- (c) 延迟且指定延迟步数为1。例如做迭代3时，可以不拿回迭代2的梯度，但必须保证迭代0、1的梯度都已拿回且用于参数更新。

**异步很香，但对一个Worker来说，只是等于W不变，batch的数量增加了而已，在SGD下，会减慢模型的整体收敛速度**。异步的整体思想是，比起让Worker闲着，倒不如让它多吃点数据，虽然反馈延迟了，但只要它在干活在学习就行。

### 2.分布式数据并行（DDP）
受通讯负载不均的影响，DP一般用于单机多卡场景。因此，DDP作为一种更通用的解决方案出现了，**既能多机，也能单机**。DDP首先要解决的就是通讯问题：**将Server上的通讯压力均衡转到各个Worker上**。实现这一点后，也就成就了DDP的核心思想：**去Server，留Worker**。

#### 2.1 主要框架
DDP与DP的实现过程类似，步骤1-4都一致，**不同的是第5步，DP是Server做AllReduce，而DDP是每个Worker做AllReduce**。实现的核心操作为**Ring AllReduce**，具体介绍见附录6.AllReduce。Ring-AllReduce通过环形拓扑结构，将数据分发到相邻的节点，从而实现高效的通信。

单卡总通讯量为$2(N - 1)\frac{\Phi}{N}$，随着$N$的增大，可以近似为$2\Phi$。全卡总通讯量为$2N\Phi$。

而对前文的 DP（Data Parallelism）来说，它的 Server 承载的通讯量是$N\Phi$，Workers 为$N\Phi$，全卡总通讯量依然为$2N\Phi$。**虽然通讯量相同，但搬运相同数据量的时间却不一定相同**。DDP 把通讯量均衡负载到了每一时刻的每个 Worker 上（**其通讯时间仅取决于逻辑环中最慢的两个 GPU 的连接，且不随GPU数量的增多而增多）**，而 DP 仅让 Server 做勤劳的搬运工。当越来越多的 GPU 分布在距离较远的机器上时，DP 的通讯时间是会增加的（**其通讯时间随GPU数量增多而增大**）。

![](Figure/parallel/ring.png)

如果一个节点上的所有 GPU 在环中彼此相邻，则该算法的功能最佳；这最小化了网络争用的量，否则这可能会显著降低 GPU-GPU 连接的有效带宽。

#### 2.2 缺点

DDP存在的问题：在N张卡进行训练，设模型参数量为M，采用全精度参数，则需要参数+梯度+优化器(Adam优化器需要存储一阶动量和二阶动量)=（4+4+8）*M空间。**占用显存空间过大！**

### 3.零冗余优化器（ZeRO）
ZeRO主要解决DP中每个GPU占用显存过大的问题，其主要思想是：**将优化器状态、梯度和模型参数进行分割，每个GPU只保存部分数据**。

ZeRO有三个阶段：
- ZeRO Stage 1：仅对优化器状态进行分割，每个GPU中仍有完整的模型参数和梯度数据
- ZeRO Stage 2：对优化器状态和梯度进行分割
- ZeRO Stage 3：对优化器状态、梯度和模型参数全部进行分割

![](Figure/parallel/zero.png)

具体的ZeRO实现过程详见文章[DeepSpeed](https://summer536.github.io/Notes/zh/posts/DeepSpeed.html)


## 二、Model Parallel


## 三、Pipeline Parallel


## 四、Tensor Parallel

## 五、Sequence Parallel


## 六、Expert Parallel





## 附录：常见集合通信算子
集合通信（Collective Communications）是一个进程组的所有进程都参与的全局通信操作，其最为基础的操作有 **发送 send**、**接收receive**、**复制 copy**、**组内进程栅障同步 Barrier** 以及**节点间进程同步(signal +wait )**，这几个最基本的操作经过组合构成了一组通信模板也叫通信原语，比如：1 对多的广播 broadcast、多对 1 的收集gather、**多对多的收集 all-gather**、**1 对多的发散 scatter**、**多对 1 的规约 reduce**、**多对多的规约 all-reduce**、**组合的规约与发散 reduce-scatter**、**多对多的 all-to-all** 等，集合通信的难点在于通信效率以及网络硬件连接拓扑结构的最佳适用。


### 1.Scatter(1分片->N分发)
Scatter 是数据的 1 对多的分发，它将一张 XPU/GPU 卡上的数据进行**分片再分发**到其他所有的 XPU/GPU 卡上，他的反向操作对应 Gather，其应用场景有：

- ReduceScatter 组合里的 Scatter 操作；
- 模型并行里初始化时将模型 Scatter 到不同的 XPU 上；

![](Figure/parallel/scatter1.png)

### 2.Gather(N分片->1合并)
Gather 是数据的多对 1 的收集，它将多张 XPU 卡上的数据收集到 1 张 XPU 卡上，他的反向操作对应 Scatter

### 3.AllGather(N分片->1合并->N广播)
AllGather 属于多对多的通信原语，具有多个数据发送者，多个数据接收者，可以在集群内**把多个节点的数据收集到一个主节点上（Gather），再把这个收集到的数据分发到其他节点上（broadcast）**，即收集集群内所有的数据到所有的节点上。可以看做 Gather + Broadcast 的操作组合，它的反向操作对应 ReduceScatter，其最应用场景有：

- AllGather 可应用于模型并行；
- 模型并行里前向计算里的参数全同步，需要用 allgather 把模型并行里将切分到不同的 XPU 上的参数全同步到一张 XPU 上才能进行前向计算。

### 4.Reduce(N分片->1规约计算)
Reduce 是数据的多对 1 的规约运算，它将所有 XPU 卡上的数据，规约（比如 SUM 求和）到 1 张XPU卡上，其应用场景有：

- AllReduce 里的 broadcast + reduce 组合里的 reduce 操作；
- ReduceScatter 组合里的 reduce 操作；
- 分布式训练 parameter server 参数服务器结构里的 master节点 broadcast 数据到 worker 节点，再从worker 节点 reduce 数据回 master 节点里的 reduce 操作；

### 5.ReduceScatter(N分片->1规约计算->N分发)
ReduceScatter 是数据的多对多的 reduce + scatter 运算，它将所有的 XPU 卡上的数据**先规约（比如 SUM 求和）到 1 张 XPU 卡上，再进行 scatter**。如下图所示，先 reduce 操作 XPU 0-3 的数据 reduce 为 A(A0+A1+A2+A3) + B(B0 + B1 +B2 + B3) + C(C0 + C1 + C2 + C3) + D(D0 + D1 + D2 + D3 ) 到一张 XPU 上，再进行分片 scatter 到集群内所有的 XPU 卡上。

![](Figure/parallel/reducescatter.png)

其应用场景有：

- ReduceScatter即可应用于数据并行也可应用于模型并行；
- 数据并行 allReduce 里的 ReduceScatter+ Allgather 组合里的 ReduceScatter 操作；
- 模型并行里在前向 allgather 后的反向计算里的 ReduceScatter；

### 6.AllReduce(N分片->1规约计算->N广播)

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

可以看到，**单个设备通信量与 GPU 数量 $N$ 无关**，通信量为：
$$
\Psi*(N-1)/N + \Psi*(N-1)/N = 2\Psi*(N-1)/N
$$

**当GPU数量足够多时，单卡总通信量趋近于 $2\Psi$，即对于每个单卡，其通信量与 GPU 数量无关。**

值得注意的是，使用张量并行加速时，分布式系统 Allreduce 的**通信速度只受限于逻辑环中最慢的两个 GPU 的连接**;（每次需要通信的数据大小仅为 $\Psi/N$，随着 $N$ 增大，通信量减少，一般小于 network bandwidth）；总结就是 Ring Allreduce 的通信速度恒定，和设备数量无关，完全由系统中GPU 之间最慢的连接决定。

## 总结



## 待更新

## 参考资料


[大模型的分布式训练框架：deepspeed](https://mp.weixin.qq.com/s/kYeNjMsesfKfoZtJPRkciA)

[深度学习常见AllReduce算法图解](https://zhuanlan.zhihu.com/p/469942194)
