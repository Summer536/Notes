---
title: Interview
date: 2025-08-16
readingTime: 300
category:
  - 笔记
tag:
  - GPU优化
# cover: /assets/images/cover3.jpg
isOriginal: true
---

# Interview

## 一. C++
C++是基础，会在面试校招生和社招两年内时考察。

### 1. 堆和栈的区别？
1. 管理方式：栈由编译器管理，堆由程序员手动管理；
2. 碎片问题：栈不存在碎片问题，堆很容易产生碎片；
3. 分配方式：栈是编译器自动分配的，堆是(new,delete/malloc,free)分配。
4. 分配效率：栈的效率比堆高；

### 2. 以下四行代码会调用到什么函数？
// A是一个C++类
A a1(1); //构造函数
A a2(2); //构造函数
a2 = a1; // 拷贝赋值，operator=
A a3 = a1; //拷贝构造函数（使用已经初始化的对象a1去初始化一个没有初始化过的对象a3，符合拷贝构造函数定义）

### 3. C++类的拷贝构造函数有哪几种调用场景？
1. 对象作为函数的参数，以值传递的方式传入函数体；（扩展，当拷贝构造函数的参数不是引用类型(const MyClass& other)，而是值类型(MyClass other)的时候，可能存在**循环调用拷贝构造函数的情况，造成栈溢出**）
```cpp
void foo(MyClass obj);   // obj 需要拷贝构造
MyClass a;
foo(a);  // 调用拷贝构造函数
```
2. 对象作为函数返回值，以值传递的方式从函数返回；（扩展，现代编译器会采用RVO(return value optimization)**返回值优化**来避免拷贝构造）
```cpp
MyClass foo() {
    MyClass obj;
    return obj;  // 返回值需要拷贝构造（可能被 RVO/NRVO 优化掉）
}
```
3. 对象需要通过另外一个未初始化过的对象进行初始化；（如上2.题中的第四个函数）

#### 3.1 C++中如何访问类的私有成员？
1. 友元函数
2. 通过目标类的成员函数访问
```cpp
#include <iostream>

class MyClass {
private:
    int privateData;
public:
    // 构造函数
    MyClass(int data) : privateData(data) {}

    // 2.类的访问器函数，可获取私有成员的值
    int getPrivateData() const {
        return privateData;
    }
    // 1.友元函数
    friend int accessPrivate(MyClass& obj);       
};

int accessPrivate(MyClass& obj) {
    return obj.privateData;
}

int main() {
    MyClass obj(42);

    obj.privateData //❌ 私有变量无法被直接访问
    std::cout << "Private data: " << obj.getPrivateData() << std::endl;//✅ 可通过类的成员函数访问
    std::cout << "Private data: "  << accessPrivate(obj) << std::endl;//✅ 可通过友元函数访问
    return 0;
}
```

### 4.哪些函数不能是虚函数？
答出两三个常见的即可，并说明为什么不能是虚函数。

虚函数的本质是通过 **虚函数表指针（vptr）+ 虚函数表（vtable）** 实现的多态；属于运行时阶段(runtime)而非编译时阶段(compile time)；虚函数必须是类的成员函数

1. **静态成员函数**：静态成员函数属于类本身，而不属于类的任何对象，没有this指针，无法实现多态；（扩展，**静态二字表面其在编译器阶段已经确定**，无法在运行时动态绑定）
2. **构造函数**（析构函数是不是虚函数？可以是）：对象创建过程中，虚函数表指针vptr还未初始化，无法实现动态绑定；（扩展，如果类要作为基类并被多态使用 → 析构函数一定要是虚函数（因为它要对父类和子类同时析构）；如果不作为基类则不需要。）
3. **友元函数**：友元函数不是类的成员函数，它只是被授予了访问类的私有成员的权限；
4. **内联函数**：同静态一样，内联函数在编译器阶段已经确定；

扩展：**虚函数和纯虚函数的区别**：虚函数是基类中用 virtual 修饰的函数，可以有函数体，子类可以选择是否重写override。**纯虚函数**是用 =0 定义的虚函数，没有函数体，**要求子类必须override实现**。

### 5. unique_ptr、shared_ptr、auto_ptr的区别？
1. unique_ptr：不支持拷贝，只支持指针所有权的转移；
2. shared_ptr：支持拷贝，内部有引用计数，最后一个 shared_ptr 析构时才释放资源；
3. auto_ptr：拷贝或赋值时，所有权会转移，被赋值的一方失去所有权，置为 nullptr。（**C++11已弃用**）


```cpp
#include <iostream>
#include <memory>

// 1. unique ptr
int main() {
    std::unique_ptr<int> ptr1;
    ptr1.reset(new int(42)); //或者std::unique_ptr<int> ptr1(new int(42)); 因为unique_ptr 的构造函数接受裸指针，但赋值必须用 reset 或初始化。
    std::unique_ptr<int> ptr2;
    ptr2 = ptr1; //❌ unique_ptr不支持拷贝
    ptr2 = std::move(ptr1); //✅ 指针的所有权随之转移 print ptr1.get()会得到 nullptr
    return 0;
}


// 2. shared ptr
int main() {
    std::unique_ptr<int> ptr1(new int(43));
    std::shared_ptr<int> ptr2;
    ptr2 = ptr1 //✅ 引用计数+1，ptr1.use_count() == ptr2.use_count() == 2
    return 0；
}
```

### 6.四种强制类型转换是什么？它们的区别？
1. **const_cast**：去掉或添加 const / volatile 限定。
2. **static_cast**：基本类型转换（**上、下行转换都可用**），但使用它实现父类转子类(下行转换)不太安全。（扩展，如果程序员可以确认该父类转子类是安全的情况下，优先用static_cast比dynamic_cast性能更佳）
3. **dynamic_cast**：仅用于**下行转换**，**安全**，失败返回 nullptr。
4. **reinterpret_cast**：用于底层类型之间的转换，例如float=>float4。

**注意：上行转换是绝对安全的**

```cpp
#include <iostream>

// 定义父类
class Shape {};

// 定义子类
class Circle : public Shape {};

int main() {
    Circle circle;// 创建子类对象

    // 向上转换：子类指针转换为父类指针，绝对安全。
    Shape* shapePtr = &circle; //不需要显式写，编译器会自动优化为 Shape* shapePtr = static_cast<Shape*>(&circle); 

    // 向下转换：父类指针转换为子类指针，程序员确保安全时可用static_cast，开销小；
    Circle* circlePtr = static_cast<Circle*>(shapePtr);

    // 向下转换：父类指针转换为子类指针，程序员无法确保安全时可用dynamic_cast，开销大；
    Circle* circlePtr = dynamic_cast<Circle*>(shapePtr);

    return 0;
}
```

### 7.const和constexper的区别？
1. const只表示**只读语义**，constexpr表示**常量语义**；
2. constexpr用作修饰编译期常量，可以在编译期提前处理以减少runtime时间；可广泛用于模版参数（因为模版参数是要在编译期确定的）
```cpp
const int a = 1 + 1;  //它是变量，在runtime时计算得出，不可传入模版参数
constexpr int b = 1 + 1; //它是常量，在编译期即可被编译器算出确定，可传入模版参数
```

### 8. shared_ptr中的引用计数用什么数据结构实现？可以用static变量来实现吗？为什么？
int*

不能！

**static修饰的变量属于类所有**，而每个shared_ptr的是同一类的不同对象，如果采用static修饰，则不同的shared_ptr的引用计数都是同一个变量，会导致计数错误。见如下例子

```cpp
shared_ptr<int> ptr1(new int(42)); //ptr1是shared_ptr类的一个对象
shared_ptr<int> ptr2(new int(43)); //同理，ptr2是shared_ptr类的另一个对象
shared_ptr<int> ptr3;

ptr3 = ptr2; //此时ptr3的引用计数 == ptr2的引用计数 == 2，ptr1的引用计数为1

//如果采用static变量来实现shared_ptr的引用计数，则ptr1、ptr2、ptr3的引用计数都是同一个变量，都为2，ptr1计数错误了。

```

### 9. shared_ptr中的引用计数可以用int类型吗？如果可以，下面的代码会发生什么？
```cpp
class shared_ptr {
    int count; //如果这里用int类型
}

shared_ptr<T> a = make_shared<T>(1); //此时a的计数为1
shared_ptr<T> b = a; //此时b的计数为2（拷贝构造函数在a的引用计数器上+1 = b的计数），a的计数还为1，但是a本该也为2
shared_ptr<T> c = a; //同理c的计数为2，a的计数为1，但是a本该也为2 
```
以上代码如果使用正确的int*,则可正确计数。因为**int*是指针，指向同一块空间，b、c计数改变时，它们的计数器为同一块buffer，a也会更着同时变化的**。


### 10. 以下代码可以如何优化？这是什么优化原理？（RAII）
我们想让编译器自动释放一些资源，这样可以让我们不用手动释放资源，避免资源泄露。

这就用到了RAII（Resource Acquisition Is Initialization）特性：在main函数中将对象初始化为一个局部变量，在使用完成后编译器会自动调用对象的析构函数，从而释放资源。

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    // 构造函数
    Resource() {
        std::cout << "Resource acquired." << std::endl;
    }

    // 析构函数
    ~Resource() {
        std::cout << "Resource released." << std::endl;
    }

    void use() {
        std::cout << "Using the resource." << std::endl;
    }
};

int main() {
    // 初始化局部变量
    Resource r = Resource();

    // 使用 resource 对象
    r.use();

    // main 函数结束时，resource 对象的生命周期结束，其析构函数会被自动调用
    // 这将释放 Resource 对象所占用的资源

    return 0;
}
//输出
//Resource acquired.
//Using the resource.
//Resource released.

```

### 11. 多个if else下都是高度重复的代码，并且调用了大量的函数参数，可以如何简化这段代码？
```cpp
void TestFun()
{
    int a, b, c;
    .....   // 巴拉巴拉的一堆变量

    if (条件1)
    {
        // 除了几个变量不同外都是相同的代码块
        // 其中使用a、b、c....一堆变量
        func(a,b,c,d,e,f,g...)
    }
    if (条件2）
    {
        // 除了几个变量不同外都是相同的代码块
        // 其中使用a、b、c....一堆变量
    }
    if (条件3)
    {
        // 除了几个变量不同外都是相同的代码块        // 其中使用a、b、c....一堆变量
    }
}
```
答：lambda表达式
```cpp
#include <iostream>
using namespace std;

void TestFun()
{
    int a = 10, b = 20, c = 30;   // 公共变量
    int d = 40, e = 50;           // 公共变量

    // 定义一个 lambda，把重复的代码放在里面
    auto lambdaFun = [&](int param) {
        // 这里可以直接用外部变量 a,b,c,d,e（因为我们用 & 捕获了） ！！！！！！只写特殊变量param即可
        cout << "a+b+c = " << (a + b + c) << endl;
        cout << "d*e   = " << (d * e) << endl;

        // 只有 param 是不同的
        cout << "param = " << param << endl;
    };

    // 不同条件下，只传不同的参数
    bool 条件1 = true;
    bool 条件2 = false;
    bool 条件3 = true;

    if (条件1) {
        lambdaFun(100);   // param1
    }
    if (条件2) {
        lambdaFun(200);   // param2
    }
    if (条件3) {
        lambdaFun(300);   // param3
    }
}
//输出
a+b+c = 60
d*e   = 2000
param = 100

a+b+c = 60
d*e   = 2000
param = 300
```

### 12. 模版函数有哪些编译方式？哪个更好？
一体化编译会增加编译时间，分离式编译会多次重复显式实例化，且多个cpp可能实例化同一模板造成维护困难；前者可能造成可执行文件膨胀（由于每个cpp文件都会导入.h头文件，头文件中由于已经实例化因此会造成重复代码的导入）。

但仍然一般**推荐一体化编译**

1. **一体化编译** ：模板类或函数的 声明 + 定义都放在头文件。调用时**编译器能直接看到完整定义，从而实例化**。最常见的方式，C++STL、NV的cutlass 等很多组件都是这样。
```cpp
// MyTemplate.h
template<typename T>
T add(T a, T b);

T add(T a, T b) {
    return a + b;
}
```

2. **分离式编译**：模板的 声明放在 .h，定义放在 .cpp。因为编译器在看到 .h 时并不知道 **.cpp 里的定义，所以必须显式实例化**需要的类型，否则会报链接错误。
```cpp
// MyTemplate.h
template<typename T>
T add(T a, T b);

// MyTemplate.cpp
#include "MyTemplate.h"
template<typename T>
T add(T a, T b) { return a + b; }

// 显式实例化
template int add<int>(int, int);
template double add<double>(double, double);
```

### 13. 模板实例化、特例化、偏特化的概念？
1. 模板实例化：编译器在看到具体类型时，根据模板生成对应的具体函数或类。
```cpp
// 函数模板
template <typename T>
T add(T a, T b) { return a + b; }

int main() {
    add(1, 2);       // 隐式实例化 -> 生成 add<int>(int, int)
    add(1.1, 2.2);   // 隐式实例化 -> 生成 add<double>(double, double)

    // 显式实例化
    template int add<int>(int, int);
}

```
2. 模板特例化：为某个特定类型提供完全不同的实现，覆盖原有的通用模板。
```cpp
// 通用模板
template <typename T>
class Printer {
public:
    void print(T val) { std::cout << val << std::endl; }
};

// 特例化：char* 的打印方式不同
template <>
class Printer<char*> {
public:
    void print(char* val) { std::cout << "C-string: " << val << std::endl; }
};
```
3. 模板偏特化：不是针对某一个完全确定的类型，而是针对 一类类型模式 提供特殊实现。

**部分参数特化**（个数上的偏特化）
```cpp
#include <iostream>

// 原始的模板定义
template <typename T1, typename T2>
class MyTemplate {
public:
    void print() {
        std::cout << "General template" << std::endl;
    }
};

// 部分参数偏特化，对第二个参数特化为 int 类型
template <typename T1>
class MyTemplate<T1, int> {
public:
    void print() {
        std::cout << "Partial specialization with T2 = int" << std::endl;
    }
};

int main() {
    MyTemplate<double, char> obj1;
    obj1.print(); // 调用通用模板的 print 函数

    MyTemplate<double, int> obj2;
    obj2.print(); // 调用部分特化模板的 print 函数

    return 0;
}
```

**参数范围的偏特化**（范围上的偏特化）
```cpp
#include <iostream>

// 原始的模板定义
template <typename T>
class MyTemplate {
public:
    void print() {
        std::cout << "General template" << std::endl;
    }
};

// 类型范围偏特化，对指针类型进行特化
template <typename T>
class MyTemplate<T*> {
public:
    void print() {
        std::cout << "Partial specialization for pointer types" << std::endl;
    }
};

int main() {
    MyTemplate<int> obj1;
    obj1.print(); // 调用通用模板的 print 函数

    int num = 10;
    MyTemplate<int*> obj2(&num);
    obj2.print(); // 调用指针类型偏特化模板的 print 函数

    return 0;
}
```

### 14. 使用模版来实现一个斐波那契数列（看看就好）
这个是个模版元函数的典型例子，利用了模版的递归性
```cpp
#include <iostream>
using namespace std;

// 1.使用static确保生成的value是属于整个类Fibonacci，而非某个对象，因此该类的所有成员可以直接访问无需每个对象都初始化这个变量
// 2.使用constexpr， 最终结果 55 在编译期就能算出来，运行时只是打印
template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template<>
struct Fibonacci<1> {
    static constexpr int value = 1;
};

template<>
struct Fibonacci<0> {
    static constexpr int value = 0;
};

int main() {
    const int result = Fibonacci<10>::value;
    cout << result << endl;
    return 0;
}
```

### 15.以下代码会调用哪些函数？讲讲C++的右值有哪些？
cpp中右值 (rvalue) 指的是：**不能出现在赋值语句左边的值**。
右值有两种类型，纯右值和将亡值。
1. **纯右值**：字面值（例：0、1、2），表达式（例：x+y），返回的对象
2. **将亡值**：属于对象，但即将被销毁，可以“窃取资源”。例如 std::move(a)、 static_cast<A&&>(a)中的a（对应下方代码的o = std::move(obj)中的obj）。

```cpp
class BigObj {
public:
    explicit BigObj(size_t length)
        : length_(length), data_(new int[length]) {
    }

    // 析构
    ~BigObj() {
     if (data_ != NULL) {
       delete[] data_;
        length_ = 0;
     }
    }

    // 拷贝构造函数
    BigObj(const BigObj& other) = default;

    // 赋值运算符
    BigObj& operator=(const BigObj& other) = default;

    // 移动构造函数
    BigObj(BigObj&& other) : data_(nullptr), length_(0) {
        data_ = other.data_;
        length_ = other.length_;
        //移动构造函数后，没有将原对象回复默认值
    }
    
    // 移动赋值函数
    BigObj& operator=(BigObj&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        data_ = other.data_;
        length_ = other.length_;
        other.data_ = nullptr;
        other.length_ = 0;
    }
    return *this;
}


private:
    size_t length_;
    int* data_;
};

BigObj func(int a, int b){
}//RVO

int main() {
   BigObj obj(1000); //构造函数
   BigObj o; //构造函数

   {
    o = std::move(obj); //移动赋值（因为o已经被创建了，因此这里不是拷贝构造函数），将obj转为右值
   }//析构o对象
   //{}表示一个作用域，离开这个作用域o马上就会被析构

   return 0;//析构ocj对象
}
```


## 二. 操作系统
这一块纯八股，知道操作系统调度算法、进程线程和虚拟物理地址即可

### 16.为什么要有虚拟地址？物理地址和虚拟地址的区别？
举个例子说明：

比如，单片机，它只有物理地址没有虚拟地址，因此如果单片机上有程序，我继续输入新的程序就会把就程序给cover掉。它没有内存抽象等的保护因此会出现这种情况，因此催生出来虚拟地址。

1. **内存抽象**：虚拟地址为程序提供了一个**抽象的内存环境**，使得程序在运行时不需要考虑物理内存地址；**程序使用的是逻辑地址，由操作系统映射到物理地址**。
2. **内存空间扩展**：通过虚拟地址技术，操作系统可以提供比物理内存更大的虚拟内存空间；这使得多个程序可以并行，每个程序拥有独属于自己的一块内存。
3. **内存保护**：**不同进程拥有不同的虚拟地址空间**，一个进程的代码和数据不能访问另一个进程的，确保了相互隔离和安全。
4. **内存管理**：操作系统可以通过分页(page)或分段(Segmentation)技术来动态的实现内存的分配和回收，提升内存利用率。

### 16.1 既然虚拟内存地址要比物理内存空间要大，那岂不是会出现物理内存不够用的情况？如何解决？
当物理内存不足时，操作系统会使用**页置换（Page Replacement）**的方法将一些不常用的内存从物理内存移动到磁盘上的交换区 (Swap) 中，腾出空间。

当这些页再次被访问时，会发生缺页中断 (Page Fault)，再把它们从磁盘加载回内存。

常见算法：

- FIFO (先进先出)：最先进入内存的页最先被换出；
- **LRU (最近最少使用)**：优先换出最长时间没被访问的页（扩展：大模型推理里面比如说kvcache等也会使用LRU进行管理）；
- LFU (最少使用频率)：优先换出使用次数最少的页。

**LRU的Leetcode一定要做，考的概率很大**

### 17. 进程和线程的区别？
**进程是操作系统资源分配的基本单位，线程是CPU调度的基本单位。**
- 进程每个进程有自己独立的虚拟地址空间、代码段、数据段和堆栈。切换开销大、通信复杂，但更稳定，适合高隔离场景；
- 同一进程内的线程共享进程的地址空间和资源，但各自有独立的栈和寄存器。切换开销小、通信简单，但容易互相影响，适合高并发计算场景。


## 三. AI-HPC基础

### 18. FP32、FP16、BF16三者在IEEE754标准下的区别？
| 类型  | 总位数 | 符号位 | 指数位 | 尾数位 | 指数偏置 | 范围 | 典型应用场景 |
|-------|--------|--------|--------|--------|----------------|------|--------------|
| FP32  | 32     | **1**      | **8**      | **23**     | 127      | ~±3.4e38 | 高精度计算、训练 |
| FP16  | 16     | **1**      | **5**      | **10**     | 15       | ~±6.5e4  | AI推理、节省内存 |
| BF16  | 16     | **1**      | **8**      | **7**      | 127      | ~±3.4e38 | AI训练、兼顾范围和效率 |


### 19. FP16与BF16的对比？
BF16更适合training，因为指数位宽，动态范围大，**减少梯度爆炸和梯度消失**，更有利于backward；

FP16更适合推理，推理只有forward，不涉及backward，因此不考虑梯度问题，只考虑精度即可。FP16**尾数位多，对精度更友好**

### 20. 写出代码计算1+(1/2)+(1/3)+(1/4)+...+(1/n)
[flaot 数据类型的一些坑（大数吃小数）](https://blog.csdn.net/xieyihua1994/article/details/106137932)

重点提取：

1. 在浮点数计算前，会先将其转为二级制的科学计数。ep.(-1)^sign * 1.f * 2^e
2. 浮点数加法需要先对齐指数位再相加, ep.0.5+0.125 = (-1)^0 * **1.0** * 2^(-1) + (-1)^0 * **0.01** * 2^(-1) = (-1)^0 * **1.01** * 2^(-1) 
3. 大数加小数时，为了对齐指数位，需要左移动1.f的小数点，当移动超过23位时，小数变成了0（例如：(-1)^0 * **0.0** * 2^(-1)），就意味着+的这个数就是0。此时会出现**大数吃小数**的情况（例如：16777216 + 1 = 16777216）

**注意，这种情况只出现在浮点数运算中，整数运算不会出现**

```cpp
#include <iostream>
using namespace std;

int main(){
    long i, n; //int表示32位的整数，long 表示更大位的整数，long long会保证至少64位的整数
    double sum;
    cin >> n;
    
    sum = 0.0;
    for(i = n; i >= 1; --i){ //这里for采用--i而不是++i就是为了防止大数吃小数的情况发生
        sum += 1.0/i; //这里必须是1.0，不能是1，因为这是浮点数除法不是整数除法
    }
    cout << sum << endl;
    return 0;
}
```
### 21. INT8和FP8在IEEE754标准下的区别？

| 类型  | 总位数 | 符号位 | 指数位 | 尾数位 | 典型应用场景 |
|-------|--------|--------|--------|--------|--------------|
| INT8  | 8      | 1      | 0      | 7      | 量化推理、存储节省、嵌入式计算 |
| FP8   | 8      | 1      | 4 或 5 | 3 或 2 | AI 推理加速、低精度训练（Edge TPU / GPU TensorCore） |

### 22. INT8和FP8的优劣势对比
1. 二者都是为模型量化而生，fp8相对于int8在精度上面不具有理论优势，因为**int8为均匀数据类型**，在值域内都是均匀的，数和数之间间隔为1，但**fp8不是均匀数据类型**，数与数之间间隔不定，但是个数一定为3个，这也导致了在数**值小的范围里面，fp8的精度优于int8，但是在数值偏大的范围里面，int8精度优于fp8**。

2. **fp8可以用来做模型训练**（例如Deepseek就使用fp8精度做的训练），并且fp8训练的模型可以**直接使用fp8进行推理**。**Int8不能用作推理**因为值域里面全是整数，没法求导，从而没法计算梯度。

3. fp8量化理论上相对int8不需要校准(calibration)，直接设置input scale和output scale为1即可量化，（注意：实际上多数fp8量化项目还是做了calibration），int8需要calibration

### 23. Attention包含哪些算子？使用pytorch搭建一下。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    ## d_model表示hidden size, d_head表示head num
    ## 单头注意力：hiddensize = head num;
    ## 多头注意力：hiddensize = head num * headsize;
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, d_head) ##将输入的hiddensize维度转为headnum维度，进行Attention计算
        self.W_k = nn.Linear(d_model, d_head) ##nn.Linear(输入维度, 输出维度, bias=True默认)
        self.W_v = nn.Linear(d_model, d_head)
        self.W_o = nn.Linear(d_head, d_model) ##Attention计算后的结果（维度为 d_head）映射回原始模型维度 d_model

    def forward(self, x):
        """
        x:    [bs, seq_len, d_model]
        mask: [seq_len, seq_len] 注意力分数是一个 seq*seq 的矩阵，因此掩码矩阵要和它shape一致
        """
        bs, seq_len, _ = x.shape

        ## QKV linear
        Q = self.W_q(x)  #[bs, seq_len, d_model] -> [bs, seq_len, d_head]
        K = self.W_k(x)  #[bs, seq_len, d_model] -> [bs, seq_len, d_head]
        V = self.W_v(x)  #[bs, seq_len, d_model] -> [bs, seq_len, d_head]

        ## RoPE
        Q, K = apply_rope(Q, K, cos, sin)

        ## Q*(K^T) / sqrt(d_head)
        attn_socres = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5) #[bs, seq_len, seq_len]
            # -2 -1表示K的倒数第一和第二维度也就是[seq_len, d_head],transpose为[d_head, seq_len]

        ## Mask
        if mask is not None:
            attn_socres = attn_socres.masked_fill(causal_mask == 0, float('-inf'))  #[bs, seq_len, seq_len]
            # causal_mask矩阵shape与attn_socres一样，需要掩码的地方值为0，不需要的地方值为1
            # masked_fill函数是将attn_socres矩阵中和causal_mask矩阵为0的位置相同的位置 将其值置为-inf
            # softmax会将-inf的值的概率算为0（这也是为什么Mask要放在Softmax前面的原因）
        
        ## Softmax
        attn_weights = torch.softmax(attn_socres, dim=-1) #[bs, seq_len, seq_len]
            # dim=-1表示对最后一个维度进行softmax，每一行（比如第 i 行）表示：第i个token对所有其他token的原始注意力分数
            # 所以要在每行内部做 softmax —— 也就是在最后一个维度（列方向）上操作。

        ## O = attn_weights * K
        output = torch.matmul(attn_weights, V) #[bs, seq_len, d_head]
        output = self.W_o(output) #[bs, seq_len, d_model]

        return output

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    d_model = 128
    d_head = 64
    seq_len = 8
    bs = 2

    model = SimpleAttention(d_model, d_head)
    x = torch.rand(bs, seq_len, d_model)

    out = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)

```


## 四. X86CPU体系结构

### 24. CPU上有哪些并行策略？











## 五. GPU体系结构与cuda编程

## 六. 大模型量化

## 七. 大模型推理

## 八. Leetcode