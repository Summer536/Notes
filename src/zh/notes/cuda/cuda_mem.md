---
title: CUDA内存分配函数介绍
date: 2026-03-24
readingTime: 300
category:
  - 笔记
tag:
  - GPU优化
# cover: /assets/images/cover3.jpg
isOriginal: false
---

# CUDA内存分配函数 

## 简介

本文将讲解CUDA中经常使用的三个高频内存分配函数：cudaMalloc、cudaMallocManaged、cudaMallocAsync。

1. 无需频繁分配显存，数据与CPU几乎无交互 => 选cudaMalloc；

2. 需要CPU和GPU共用数据，少写代码、简化编程 => 选cudaMallocManaged + 2MB大页表；

3. 频繁分配、释放内存（比如AI批次推理、训练）=> 选cudaMallocAsync，遵守32MB对齐。

<!-- more -->

## 一、cudaMalloc（最常用的GPU显存分配方式）

在初学 CUDA 时，我们首先接触的GPU内存分配函数便是 cudaMalloc ，该函数是CUDA中最通用、最基础的内存分配的Rumtime API。（这里简单介绍一下，Runtime API是指给程序员封装好的易用接口，相对应的是Driver API；后者一般只在深度优化时才用到。可以类比于C语言与汇编语言。）

### 1.1 cudaMalloc 特性

- **访问权限**： 只可用于GPU，CPU操作时会出现经典的SegmentFault。（程序访问了“它不该访问的内存”导致崩溃）且必须使用cudaMemcpy来手动搬运数据（CPU与GPU之间来回传递）

- **适用场景**： 纯GPU计算，数据不需要频繁在CPU与GPU之间切换。

- **核心优势**： 简洁直观，且内存固定于GPU上不会发生内存访问错误。

### 1.2 代码示例

```cpp
// 1. 定义分配内存的大小（size_t是c++中的无符号整数类型，专用于表示大小或内存容量）（c++标准规定sizeof运算符返回size_t类型）
const size_t data_size = 1024 * sizeof(float);
float *h_data, *d_data;

// 2. CPU端分配内存，并初始化数组
h_data = (float*)malloc(data_size);
for (int i = 0; i < 1024; ++i){
   h_data[i] = (float)i;
}

//3. GPU端分配内存，并拷贝数据
cudaMalloc((void**)&d_data, data_size);
cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

//4. 执行kernel
kernel<<<1, 1024>>>(d_data);

//5. 执行结果返回CPU
cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost);

//6. 清理内存
cudaFree(d_data);
free(h_data);
```

## 二、cudaMallocManaged（CPU与GPU共用的Unified mem（统一内存））

cudaMallocManaged 可以直接为CPU与GPU的“共用指针”来分配内存，其内存对CPU、GPU都可见，无需再使用cudaMemcpy来进行内存来回拷贝。但是，由于涉及到底层的内存管理系统，使用该方法分配的内存极易由于“缺页中断”、“TLB Miss”等原因造成性能大幅下降。

### 2.1 内存管理核心概念

要真正用好cudaMallocManaged，必须先了解GPU内存管理的底层逻辑即**MMU**、**TLB**以及**页表**的协同工作。

#### 2.1.1 页表、TLB、MMU的概念

现代CPU、GPU内存都采用虚拟内存机制，即程序员写代码时访问的都是虚拟地址，而数据真正存储的位置是物理地址。页表、TLB、MMU的核心作用就是完成虚拟地址到物理地址的映射。（vLLM的 PageAttention也是此概念）

- **页表（Page Table）**：是虚拟地址与物理地址的“映射字典”，每一条记录对应于一个“内存页”，明确标注者某段虚拟地址对应哪段物理地址。

- **TLB（Translation Lookaside Buffer）**：缓存一些最近常用到的页表映射，下次再用到之前用过的页表时查阅速度会加快。类似cache。

- **MMU（Memory Management Unit，内存管理单元）**：负责执行虚拟地址到物理地址的转换，全程依赖页表和TLB，由它来决定内存访问速度。

#### 2.1.2 MMU工作流程
程序访问内存的全过程，即是三者协同工作的过程。

1. 程序发起内存访问请求；

2. MMU首先查询TLB， 判断该虚拟地址的映射是否在缓存中；

3. A. 如果**在缓存中（TLB Hit）**，则直接获取对应的物理地址，快速访问数据，全程无需驱动介入，**速度最快**；

4. B. 如果**未在缓存中（TLB Miss）**，则MMU会遍历整个页表，查询对应的物理地址；同时将该映射缓存至TLB中；

5. C. 如果遍历页表都没找到（**页表映射无效**），则会触发**缺页中断**，这也是cudaMallocManaged**性能损耗的核心原因**。


#### 2.1.3 缺页中断（Page Faulting）及其性能影响

缺页中断是cudaMallocManaged性能不稳定的核心因素。

**缺页中断的本质**：当MMU查不到虚拟地址对应的物理地址（页表映射无效）时，说明数据不在当前处理器（CPU/GPU）能直接访问的物理内存/显存上，需要**CUDA驱动介入处理**，而这个驱动介入进行数据搬运的情况就是缺页中断。

**以一个GPU访问CPU内存数据的实际场景来看看缺页中断的触发条件和处理过程：**

1. GPU要访问某个虚拟地址用于计算；

2. MMU查询TLB未命中，再查询页表，发现该虚拟地址的映射“无效”，说明**数据不在GPU显存中**；

3. 触发GPU硬件中断，**GPU暂停当前所有计算任务，等待CUDA驱动处理**；

4. 驱动通过PCIe总线，将该虚拟地址对应的data从CPU**内存搬运**到GPU显存中；

5. 驱动更新GPU页表， 将该虚拟地址的映射**标记为“有效”**，并将映射指向刚刚搬运过来的显存物理地址；

6. 驱动通知GPU，**GPU恢复计算**，继续访问数据。

### 2.2 cudaMallocManaged 的特性
基于上述介绍，很容易理解cudaMallocManaged有以下的特性：

- **访问权限**： CPU与GPU共用一个虚拟地址指针，无需手动搬运数据，驱动会自动处理

- **底层机制**： 全靠缺页中断时的GPU驱动自行拷贝数据

- **使用场景**： 需要CPU和GPU共用数据，少写代码，简化编程

- **核心优势**： 无需手写cudaMemcpy来回拷贝数据

- **核心痛点**： 数据量大时会多次触发缺页中断，导致性能忽高忽低。

### 2.3 cudaMallocManaged的深度解析与优化

#### 2.3.1 与cudaMalloc的对比
上文讲过，cudaMallocManaged的核心是“动态换页”，依靠GPU驱动来处理缺页中断自动搬运数据，实现CPU/GPU共用指针，但也因此带来的性能开销。

而反观cudaMalloc分配的是纯GPU内存，从分配的那一刻起，GPU的页表中的虚拟地址和物理显存地址就已经绑定死了。所以它不用处理缺页中断，不用动态更新页表，没有额外的隐式搬运开销，性能更稳定。

#### 2.3.2 如何提升cudaMallocManaged的性能

其性能瓶颈就出现在页表与缺页中断上，具体来说：

- TLB容量有限，无法存储过多的页表映射。尤其是使用64KB小页的时候，页表条目会急剧增多，很容易导致TLB Miss频繁发生， MMU只能反复遍历页表来查找物理地址，额外增加了开销

- 大数据量下，64KB小页表会触发数万次甚至几十万次缺页中断，GPU大部分时间中都在等待驱动操作，导致性能严重损耗

而针对上述问题，一个很直接的办法是进行**大页优化**（但CUDA的官方手册里说，GPU的页表大小是固定的2MB，？？？），**增大页大小**，**减少页表条目**以及**缺页次数**，就能从根源上缓解性能瓶颈。例如搬运1GB的数据，采用64KB页需要16384次缺页中断，而2MB则只需512次，驱动介入频率下降，性能稳定性更佳。

### 2.4 代码示例

```cpp
// 1. 定义要分配的内存大小（示例：2MB大页，必须对齐）
const size_t ALIGN_2MB = 2 * 1024 * 1024;
float *managed_data;

// 2. 分配统一内存
cudaMallocManaged((void**)&managed_data, ALIGN_2MB);

// 3. 设置大页优化和内存偏好（进一步提升性能）
//（可选）cudaMemAdvise = 告诉 CUDA 这块内存怎么用更快，可设置多条建议（只是建议，不会强制）
cudaMemAdvise(managed_data, ALIGN_2MB, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId); //建议：首先位置为CPU
cudaMemAdvise(managed_data, ALIGN_2MB, cudaMemAdviseSetReadMostly, cudaCpuDeviceId); //建议：这块内存做只读优化（读取多，写入少）

//（可选）cudaMemPrefetchAsync可提前把数据搬到GPU上
//cudaMemPrefetchAsync(managed_data, size, device_id);

// 4. CPU端直接访问统一内存（不用手动搬数据，省代码）
for (int i = 0; i < ALIGN_2MB / sizeof(float); i++) {
   managed_data[i] = (float)i; // CPU直接初始化数据
}

// 5. GPU端直接访问同一个指针（不用cudaMemcpy）
kernel<<<1, 1024>>>(managed_data);
cudaDeviceSynchronize(); // 等待GPU执行完成，避免数据读取异常

// 6. CPU端直接读取GPU修改后的数据
printf("GPU处理后的数据：%f\n", managed_data[0]);

// 7. 清理资源（只需要释放统一内存，不用分别释放CPU和GPU内存）
cudaFree(managed_data);
```

## 三、cudaMallocAsync（针对流式的异步高效的内存池分配）

cudaMallocAsync是一个异步内存分配函数，用于解决“高频分配/释放内存”的痛点，例如AI训练、推理时，要频繁的给每一批数据分配内存，用它能节省不少开销，同时还能减少显存碎片化。

### 3.1 异步与内存池

#### 3.1.1 异步
异步的概念不局限于CPU和GPU完全独立，各干各的。cudaMAllocAsync的“异步”指的是CPU侧无阻塞，但GPU侧会受到流上下文的严格约束，具体为：

- CPU侧无阻塞：调用cudaMAllocAsync后，CPU不需要等待GPU完成内存池的内存切分操作，它直接执行后续的代码，不会被内存分配操作卡住，可大幅提升整体效率；

- 流上下文约束：分配的内存会和调用时绑定的CUDA流进行绑定，必须等待该流中“前面的操作（例如kernel执行、内存拷贝等）”全部完成后，这块内存才能被使用；同时，内存的释放、回收也必须和分配时的流保持一致，不能随意切换流，否则会触发segmentFault。

#### 3.1.2 内存池
内存池是指预先分配的内存的集合，可以在后续的内存分配中重新使用。内存池的使用与回收有以下几个点需要注意：

- **32MB对齐强制要求**：内存池的大小、起始地址，必须是32倍的整数倍；从池中分配的内存块，建议也按照32MB对齐。未对齐的后果为：要么触发cudaErrorInvaildValue错误，要么驱动自动补全对齐从而增加性能开销。

- 主动回收方式：测试、调试时，可通过```cudaMemPoolTrimTo(pool,0)```函数，强制回收内存池中的所有闲置内存，方便排查内存相关问题。

- 被动回收：当GPU显存不足时，驱动会立即回收闲置内存；当GPU长时间处于闲置状态，驱动也会主动回收；当内存池解绑、进程退出时，会必然回收所有闲置内存。

此外，内存池的分配方式也分为隐式内存池和显式内存池：

- **默认（隐式）内存池**：无需显式创建而实际存在的内存池，程序中可使```cudaDeviceGetDefaultMemPool(&pool, device);```这个API来检索设备的默认内存池。设备默认内存池中进行的内存分配是不可迁移的，这些内存分配始终可以从该设备访问（但其他设备不行，比如CPU，或其他GPU；但是，其他GPU可以通过```cudaMemPoolSetAccess```来修改访问权限，并通过```cudaMemPoolGetAccess```进行查询。CPU是没有任何办法的）。此外，设备的默认内存不支持IPC（Inter-Process Communication（进程间通信））功能。

- **显式内存池**：应用程序可以通过调用```cudaMemPollCreate```API来创建一个内存池，即显式内存池（显式内存池支持IPC，且其主要应用场景也是IPC）。该内存池只能用于设备内存的分配，具体内存分配所将驻留的设备需要在属性结构中指定，可见下方代码：
   ```cpp
   // create a pool similar to the implicit pool on device 0
   int device = 0;
   cudaMemPoolProps poolProps = { }; //内存池结构体
   poolProps.allocType = cudaMemAllocationTypePinned; //分配固定(pinned)的设备内存，它是底层可导出、可共享的物理内存，是支持IPC的关键
   poolProps.location.id = device;
   poolProps.location.type = cudaMemLocationTypeDevice;

   cudaMemPoolCreate(&memPool, &poolProps)); //创建内存池
   ```

#### 内存池的物理页面缓存及释放
程序中在调用```cudaMAllocAsync```时，如果内存池中的空余内存不足，CUDA驱动将从操作系统内核态分配更多的内存；在调用```cudaFreeAsync```时，驱动会将这些内存返回池中（因此内存池会越来越大），后续再次使用```cudaMAllocAsync```时会重新使用这部分内存。
**默认情况下，在事件、流、或设备上的下一次同步操作期间，内存池中未使用的内存会全部返回到操作系统中。**

可见，默认情况下，如果再次使用内存池时驱动会重新调用系统内核态来给空白的内存池分配内存。这会带来极大的操作开销。**一个较好地解决办法是：为内存池设置一个阈值，在事件、流、或设备上的下一次同步操作期间，内存池会保留阈值以下的内存，只将超出的部分交给系统回收**。这样下次程序再次从内存池中拿取内存时，内存池中会有一定的内存供其使用，减少了驱动开销的次数。

### 3.2 cudaMallocAsync的特性

cudaMallocAsync主要使用异步+内存池两大特性，有如下几个特点：

- **访问权限**： 仅GPU可见（且绑定流）。

- **开销极低**：分配、释放内存的速度比cudaMalloc快一个数量级，尤其适合高频内存操作场景，能大幅降低内存管理的时间成本。

- **减少显存的碎片化**：采用内存池复用机制，无需频繁向GPU系统申请、释放内存，从根源上避免了内存碎片化，提升GPU内存利用率。

- **CPU侧无阻塞**：调用cudaMallocAsync后，CPU侧无需等待GPU操作，可执行后续代码，提升整体程序的执行效率。

- **适用场景**：高频内存操作。

### 3.3 代码示例
```cpp
// 1. 定义32MB对齐常量
const size_t ALIGN_32MB = 32 * 1024 * 1024;

// 2. 创建内存池并设置释放阈值
cudaMemPool_t pool;
cudaMemPoolCreate(&pool, nullptr);//该行代码应替换为3.1.2中的显式内存池创建代码
size_t releaseThresh = ALIGN_32MB;
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &releaseThresh);//设置内存池释放阈值
cudaDeviceSetMemPool(0, pool);//设为设备0的内存池

// 3. 创建CUDA流（和内存池绑定，避免冲突）
cudaStream_t stream;cudaStreamCreate(&stream);

// 4. 异步分配内存（绑定流，CPU不用等）
float* d_data;
cudaMallocAsync(&d_data, ALIGN_32MB, stream);

// 5. 执行Kernel（使用分配的内存，和流绑定）
kernel<<<grid, block, 0, stream>>>(d_data);

// 6. 异步释放内存（绑定流，和分配对应）
cudaFreeAsync(d_data, stream);

// 7. 主动回收闲置内存
cudaMemPoolTrimTo(pool, 0);

// 8. 清理资源（按顺序销毁流和内存池）
cudaStreamDestroy(stream);
cudaMemPoolDestroy(pool);
```

## 总结

| 函数 | 类型 | 访问 | 核心机制 | 优势 | 痛点 | 场景 |
|------|------|------|----------|------|------|------|
| cudaMalloc | Device Memory | GPU | 固定设备内存（无迁移） | 稳定高性能 | 手动拷贝 | 纯GPU计算 |
| cudaMallocManaged | Unified Memory | CPU+GPU | 按需迁移（page fault + migration） | 编程简单 | 迁移开销 | CPU/GPU交互 |
| cudaMallocAsync | Memory Pool | GPU | 内存池 + stream-ordered | 低开销、复用 | 依赖stream语义 | 高频分配 |

## 参考资料

- [CUDA内存分配之cudaMalloc/cudaMallocManaged/cudaMallocAsync详解](https://mp.weixin.qq.com/s/-V7cCt51EE_e0O68AaWt0g)
- [【CUDA编程】流式有序内存分配（Stream Ordered Memory Allocator）](https://zhuanlan.zhihu.com/p/677268397)
