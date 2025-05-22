---
title: Inference服务的一些常见问题
date: 2025-05-22
readingTime: 600
category:
  - 笔记
tag:
  - GPU优化
  - 大模型
# cover: /assets/images/cover3.jpg
isOriginal: true
---

# Inference服务的一些常见问题
## 简介
记录一些常见的推理服务问题。

<!-- more -->

## 一. 大模型推理为什么要分为prefill和decode两个阶段？必须分开吗？

### 1. 什么是 Prefill 阶段？
在大语言模型的推理中，Prefill 阶段 是指处理输入提示（prompt）部分的过程：
- 输入：用户提供的初始 prompt（例如“讲个故事”）
- 输出：生成与 prompt 对应的 key/value 缓存（KV Cache），并输出第一个预测 token 的 logits
- 特点：
  - 输入长度较长（可能几百到几千 tokens）
  - **是一个密集的 attention 计算过程（可并行）**
  - 一次性处理整个 prompt，不产生新 token，只是准备 KV Cache，为后续 decode 做准备

### 2. 什么是 Decode 阶段？
Decode 阶段 是从第一个预测 token 开始，逐步生成后续 token 的过程：

- 每次只预测一个 token（或少量 token）
- 使用之前缓存的 key/value（KV Cache）加速 attention 计算
- 迭代进行，直到达到最大长度或遇到终止符（如 EOS）
- 特点：
  - **每步只处理一个 token（意味着对于同一个序列，无法并行）**
  - 利用 KV Cache 实现高效 attention

![](Figure/inference/static_batch.png)

### 3. 为什么需要分开 Prefill 和 Decode 两个阶段？
1. 计算模式不同，**前者可并行，后者不可并行**
2. KV Cache 管理优化，前者生成整个prompt的KV Cache，后者在每一步更新KV Cache（添加新生成的那个token到kvcache中）
3. 吞吐 vs 延迟的权衡，Prefill 关注的是整体延迟（prompt 越长越耗时），Decode 关注的是生成每个 token 的响应时间（即“首字延迟”和“逐字延迟”）
4. 批处理优化（Batching）
- 在服务端，多个请求可以共享 prefill 阶段（batched prefill，这个和连续批处理不一样，Batched Prefill 就是将多个用户的 prompt 同时进行预处理（即同时运行它们的 attention 计算），从而提升 GPU 利用率和整体吞吐量。）
- Decode 阶段则更难合并，因为每个请求生成的 token 序列不同（decode阶段使用的是连续批处理技术，具体见下方介绍）

**vLLM、TensorRT-LLM、DeepSpeed、HuggingFace Transformers** 等 框架都会显式地将推理划分为 prefill 和 decode 阶段，以提升效率。


## 二. 介绍一下batch、batch size、以及动态批(dynamic batching)处理技术？

### 1. 什么是batch？
在传统的深度学习推理中，比如图像分类任务：所有请求先被收集起来达到固定 batch size后统一处理，处理完这个 batch 后再处理下一个。这个收集起来的任务就叫做一个batch。

在大模型推理中，**batch 是多个用户请求的集合，每个请求包含一个或多个 prompt**。也就是说一个batch中会包含不同用户发来的多个序列。

### 2. 什么是batch size？
Batch size 是每次处理请求的序列数量。

例如下图，我们将四个用户的请求收集起来，组成一个batch，然后进行推理。此时 **batch size = 4**。
如果我们设置batch size = 3，那么它只会收集前三个用户的请求然后处理，处理完成后第四个请求组成一个新的batch开始处理。
![](Figure/inference/batchsize.png)

这里注意，一个batch中的不同序列可以有不同的长度。而不同长度也同时造成了GPU并行处理的困难。

为了统一 batch 内所有序列的长度，通常会对较短的序列进行 padding（比如用 token ID = 0 填充）
这样整个 batch 变成一个矩形矩阵（batch_size × max_seq_len），然后送入GPU进行并行处理。

Padding 带来的问题：
- 浪费计算资源（对 padding token 的 attention 计算无意义）
- 占用更多内存（KV Cache 也必须为最长序列预留空间）

为解决这一问题，引入了PagedAttention（页式注意力缓存）、DynamicBatching（动态批处理）技术等新技术。

### 3. 什么是动态批(dynamic batching)处理技术？
上面第2点讲到，在decoder阶段，GPU会并行处理一个batch中的多个序列，这些序列生成回答的长度是不同的。如果一个batch中所有序列都生成完了，GPU才会处理下一个batch。
![](Figure/inference/dynamic_batch.png)

那么举一个极端的例子，假如batchsize = 4，我们同时处理4个序列的推理生成，其中3个序列只回答了50个token，而有1个序列回答了5000个token。那么GPU必须等待这一个序列生成完毕，才能接受新的batch也就是4个新的序列。这就造成了极大的资源浪费。

动态批处理的本质就是让**GPU在处理一个batch的过程中，不断接受新的请求，并根据请求的长度动态调整batch size**。
1. 用户请求不断到来
2. 推理引擎维护一个“正在运行”的 batch
3. 新请求会被尝试添加进当前 batch（如果还有空间）
4. 如果 batch 已满或达到一定时间窗口，则启动一次推理
5. 在推理过程中，部分序列可能会完成（输出结束）
6. 完成的序列位置可以被新请求替代 → 形成新的 batch
7. 这个过程不断循环，保持 GPU 高利用率

![](Figure/inference/dynamic_batch2.png)

上图显示了通过连续批处理技术连续完成 7 个序列的推理情况。左图显示了第一次迭代后的批次，右图显示了几次迭代后的批次。每当一个序列发出终止 token 时，我们会将一个新的序列插入其位置（例如序列 S5、S6 和 S7），这样 GPU 无需等待所有序列完成即可开始处理新的序列，从而实现更高的 GPU 利用率。

## 三. 推理引擎一般都可以设置哪些参数(以vLLM为例)？

✅ 1. --host 和 --port
作用：指定 HTTP 服务监听的地址和端口
```bash
--host 0.0.0.0 --port 8000
```

✅ 2. --max-model-len（最大序列长度）

控制模型能处理的最大 prompt + output token 数量,默认值通常为 4096 或根据模型设定
长 prompt 会占用更多显存和计算资源
```bash
--max-model-len 8192
```

✅ 3. --max-num-seqs（最大并发请求数）

**控制同时进行 decode 的最大序列数量（即 batch size 上限）**
影响吞吐和延迟平衡
```bash
--max-num-seqs 1024
```

✅ 4. --max-prefill-tokens（prefill 阶段最大 token 总数）

控制单个 prefill batch 中所有 prompt 的总 token 数
避免因长 prompt 导致 prefill 耗时过长或内存爆炸
```bash
--max-prefill-tokens 8192
```

✅ 5. --max-batched-token-mem（最大 batch token 显存）

控制分配给 KV Cache 的最大显存大小（单位是 token 数量）
可防止多个长 prompt 同时进入导致 OOM
```bash
--max-batched-token-mem 8192
```
✅ 6. --swap-space（交换空间）

指定临时存储 KV Cache 的 CPU 内存大小
当 GPU 显存不足时，部分序列会被“换出”到 CPU
```bash
--swap-space 10
```

✅ 7. --gpu-memory-utilization（GPU 显存利用率）

控制用于 KV Cache 的显存比例（默认 0.9）
设置太大会导致 OOM，太小则浪费资源
```bash
--gpu-memory-utilization 0.9
```

✅ 8. --max-input-length（最大输入长度）

控制每个请求中 prompt 的最大 token 数
防止用户提交超长 prompt 导致服务不稳定
```bash
--max-input-length 8192
```

✅ 9. --max-output-length（最大输出长度）

控制每个请求最多生成多少个 token
防止无限生成或长时间占用资源
```bash
--max-output-length 1024
```

## 四. 如何针对不同推理场景（单人、多用户、单卡、多卡）设置相应的参数最大化利用GPU性能？
待更新