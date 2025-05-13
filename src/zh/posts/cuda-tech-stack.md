---
title: CUDA技术栈
date: 2025-05-08
readingTime: 60
category:
  - CUDA
  - 笔记
tag:
  - 技术
  - GPU编程
---

# CUDA技术栈知识点

## CUDA基础
- CUDA核心知识
  - 必须要会写CUDA，面试的时候手撕一定会存在
  - 算子的手撕代码通常有：
    - reduce
        - [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)
        - [Github代码](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/README.md)
        - [LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/reduce)
    - 矩阵乘（八股+可能手撕）
        - [深入浅出GPU优化系列：GEMM优化（一）](https://zhuanlan.zhihu.com/p/435908830)
        - [深入浅出GPU优化系列：GEMM优化（二）](https://zhuanlan.zhihu.com/p/442930482)   
        - [CUDA乘终极优化指南](https://zhuanlan.zhihu.com/p/410278370)
        - [CUDA SGEMM矩阵乘法优化笔记—从入门到cublas](https://zhuanlan.zhihu.com/p/518857175) 
        - [LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/hgemm)
    - [x] softmax
        - [LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/softmax)
    - [x] RMSNorm 
        - [LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/rms-norm)
    - [x] layernorm
    - [x] transpose
  - **前两个算子**要熟练掌握

- 面试经典问题
  - 如何解决bank conflict？
  - 写算子的时候如何进行roofline分析？
  - compute bound还是memory bound的判断？
  - 算子的fusion策略？
  - tiling策略的优化？

- [x] Flash Attention深入理解
  - 目前有V1、V2、V3三个版本
  - 一般考察八股文，某些组会要求手撕v1和v2（v3因复杂度高不太可能）
  - 推荐熊猫视频的讲解资料
  - 从naive softmax到safe max，再到online softmax，最后到flash attention的发展路径

- 编译链路
  - CUDA到PTX到SASS这条链路至少得了解基本原理

## Cutlass框架
- Cutlass（考察难度较高）
  - 每一代芯片的tensorcore实现
  - cute、swizzle、ldmatrix的用法
  - 单精度、半精度矩阵乘的优化技术
  - hopper架构的TMA、Wgmma以及fp8的用法

## NVIDIA基础库
- cuBLAS和cuDNN
  - 这些库的基本用法和接口
  - 常用函数和性能优化方法

## 性能分析工具
- Profiler工具掌握
  - Nsight System（Nsys profile）：系统级性能分析
  - Nsight Compute（ncu）：内核级性能分析
  - 如何分析和解读性能报告

## NVIDIA芯片架构
- NV芯片架构的发展史
  - 为什么TensorCore会发展到现在的形态？
    - [Tensorcore介绍](https://github.com/chenzomi12/AISystem/tree/main/02Hardware/04NVIDIA)
  - 从Volta到Turing到Ampere再到Hopper再到Blackwall的演进
    - [Zomi视频](https://www.bilibili.com/video/BV1mm4y1C7fg?spm_id_from=333.788.videopod.sections&vd_source=f058beebb64c488b55915da416ee6086)
  - 两大发展方向：
    1. 计算算力提升
    2. 访存的加速
  - Hopper架构上加入TMA（张量内存访存单元）加速tensorcore的访存

## 内存访问层级
- 内存访存层级理解
  - Host memory到HBM到L2到L1到寄存器等访存流程
  - 各级缓存的特点和应用
  - 数据搬运优化技术

## 学习资源
- 重点学习谷歌HPC书签下的GitHub仓库
- NVIDIA开发者文档
- Cutlass官方文档和示例 