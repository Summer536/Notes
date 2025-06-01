---
title: CPP八股
date: 2025-06-01
readingTime: 600
category:
  - 八股
tag:
  - CPP
# cover: /assets/images/cover3.jpg
isOriginal: true
---

# CPP八股
## 简介
记录一些常见的CPP八股问题。

<!-- more -->

## 一. C++基础部分

### 1. 指针和引用

指针是变量，存储的是另一个变量的地址。引用是一个变量的别名。
- 指针可以为空，引用不能为空。
- 指针本身占用内存，引用不占用内存。
- 指针需要使用解引用操作符*来访问其指向的值，引用不需要。
- 指针指向对象可变，引用指向对象不可变（绑定了就不能变）。

### 2.数据类型

整型：char, short, int, long, long long
- short int: 2字节
- int: 4字节
- long int: 4字节
- long long int: 8字节

头文件<climits>定义了符号常量，如INT_MAX, INT_MIN, LONG_MAX, LONG_MIN, LLONG_MAX, LLONG_MIN等。表示整型变量的最大值和最小值。

无符号整型：表示非负整数，可增大变量范围，但只能表示正数。内存大小与上面对应的整型相同。（只是把第一位即表示正负的符号位去掉，换成表示大小的位）
unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long
- unsigned short int: 2字节
- unsigned int: 4字节
- unsigned long int: 4字节
- unsigned long long int: 8字节


### 3.const
1. 基本用法：const修饰变量，表示该变量不可修改。
2. - 常量指针（底层const；指向常量的指针）：const int *p = &a; 表示p指向的值不可修改，但p可以指向其他地址。
    - 指针常量（顶层const；指针是一个常量）：int *const p = &a; 表示p指向的值可以修改，但p不能指向其他地址。
![](Figure/cpp/const.png)

3. 常量引用：const int &a = 10; 表示引用的值是常量，不能通过引用修改。
4. 常量成员函数，表示改函数不会修改对象的成员变量，允许const对象调用。
    ```cpp
    class MyClass {
    public:
        void func1() const { /* OK */ }
        void func2()      { /* OK */ }
    };

    const MyClass obj;
    obj.func1(); // ✅ 正确：const 对象可以调用 const 成员函数
    obj.func2(); // ❌ 错误：不能调用非 const 成员函数
    ```
5. 常量对象，表示该对象的成员变量不可修改。
6. 在函数参数中使用const修饰参数，表示该参数不可修改。
    ```cpp
    void func(const int &a) {
        a = 10; // ❌ 错误：不能修改 const 参数
    }
    ```
7. 在函数返回值中使用const修饰返回值，表示该返回值不可修改。

### 4. constexpr和const的区别
上段解释了const的用法，这里我们引入constexpr并介绍二者的区别。
首先介绍一下编译期和运行期
- **编译期**：是指代码被编译成可执行文件的过程中的阶段 。
在这个阶段，编译器会对源代码进行词法分析、语法分析、语义检查、优化和生成目标代码等操作。
所有**在编译期间就能确定的值或行为，称为编译期常量或编译期计算**。
    ```cpp
    int a = 5 + 3; // 5+3 是编译期就能计算出来的结果：8
    ```
- **运行期**：是指代码被编译成可执行文件后，在运行时执行的过程。
在这个阶段，程序会根据输入数据和运行时环境，动态地执行代码。
所有**在运行时才能确定的值或行为，称为运行期常量或运行期计算**。
    ```cpp
    int x;
    std::cin >> x;
    int b = x + 1; // x 的值在运行时才确定，所以 b 是运行时变量
    ```
1. constexpr和const的核心区别：**const可以定义编译期和运行期常量，而constexpr只能定义编译期常量**。
2. constexpr变量：在复杂系统中可能无法分辨一个变量是不是常量表达式（**常量表达式是 C++ 中那些在编译期就能被完全求值的表达式，它们的值在程序运行前就已经确定，用于支持编译期计算、模板元编程、静态检查等高级特性**），可将变量声明为constexpr，编译器会进行检查，确保该变量在编译期就能确定。
3. constexpr函数：是指能用于常量表达式的函数，函数的所有返回类型和形参都是字面类型，且函数体只有一条return语句。（有点像inline函数，可以在编译阶段展开）

constexpr的好处：
- 为一些不能被修改的数据提供保障，它们如果写成变量会有被意外修改的风险，写成constexpr后，编译器会进行检查，确保该变量在编译期就能确定。
- 提高效率：constexpr函数在编译时会被展开，避免了函数调用的开销。
- 相较于宏，没有更多的开销，且有类型安全检查更加可靠。

### 5. Volatile
**volatile 是 C++ 中一个类型修饰符（type qualifier） ，用于告诉编译器：该变量的值可能会在程序不可控的情况下被改变 ，因此不能对它进行某些优化。**（与const绝对对立）

case：
```cpp
while (flag == 0) {
    // 等待 flag 被外部设为 1
}
```
如果flag 不是 volatile，编译器可能认为它的值不会变，就将其优化为：
```cpp
if (flag == 0)
    while (true);  // 死循环
```
如果flag 是 volatile，编译器不会优化，会直接执行：
```cpp
while (flag == 0) {
    // 等待 flag 被外部设为 1
}
```
![](Figure/cpp/volatile.png)


### 6. Static
**Static 的本质(静态本质)是：“延长生命周期” 或 “限制访问范围” 或两者兼具，具体取决于使用场景。** 
1. 静态变量：使用static修饰的变量为静态变量，它的生命周期为程序运行期间，且只初始化一次。
2. 静态函数：使用static修饰的函数为静态函数。
    - 一个类中的静态函数(即静态成员函数)只能访问该类的静态成员变量而不可访问非静态成员变量或函数。
    - 静态函数不能被声明为虚函数。
    - 静态函数不能被声明为inline函数。
    - 静态函数不能被声明为constexpr函数。
    - 静态函数不能被声明为mutable函数。
    - 静态函数不能被声明为volatile函数。
    - 静态函数不能被声明为extern函数。
3. 静态成员变量：使用static修饰的成员变量为静态成员变量。
所有类的对象共享同一个静态成员变量，且只初始化一次。静态成员函数必须外部定义，以便为其分配空间。
    ```cpp
    class ExampleClass ‹
    public:
    static int staticVar; // 靜态成员变量声明
    ｝；
    //1 静态成员变量定义
    int ExampleClass: :staticVar = 0;
    ```
4. 静态局部变量：在函数内部使用static修饰的变量为静态局部变量，它的生命周期为程序运行期间，但它只对函数内可见。

![](Figure/cpp/static.png)

### 7. define、typedef（using）、inline的区别
1. define：宏定义，是预处理器处理的，在编译前会进行文本替换，没有类型检查，可能会导致一些意想不到的问题。
    ```cpp
    #define MAX(a, b) ((a) > (b) ? (a) : (b))
    ```
2. typedef：类型别名，是编译器处理的，有类型检查，可以提高代码的可读性。
    ```cpp
    typedef int MyInt;
    typedef unsigned long ulong;
    ```
    c++11 引入了using，推荐使用using定义类型别名。
    ```cpp
    using MyInt = int;
    using ulong = unsigned long;
    ```
3. inline：内联函数：告诉编译器尝试将函数调用直接替换为函数体内容 ，减少函数调用开销；但是**这只是对编译器的一个“建议”，编译器可能会忽略。**
    - 常用于短小精悍(不能存在循环语句、不能存在过多的条件判断语句)、频繁调用的函数；
    - 可以在头文件中多次定义而不会违反 ODR（One Definition Rule）；
    - 有类型检查，比宏更安全；
    ```cpp
    inline int add(int a, int b) {
        return a + b;
    }
    ```
![](Figure/cpp/define.png)

### 8. new和malloc函数区别：
**new（+delete） 是 C++ 运算符，负责内存分配和构造对象，语义完整且类型安全，底层可能调用类似 malloc 的机制；**

**malloc（+free） 是 C 语言标准库函数，只负责分配内存，语义不完整且类型不安全，底层直接调用系统调用。**

使用示例对比：
1. new:
    ```cpp
        int* p1 = new int;           // 分配一个 int，并调用其构造函数（如果是类）
        int* p2 = new int[10];       // 分配一个 int 数组
        MyClass* obj = new MyClass(); // 分配并调用构造函数
    ```
2. malloc:
    ```cpp
    int* p1 = (int*)malloc(sizeof(int));         // 分配一个 int 的内存
    int* p2 = (int*)malloc(10 * sizeof(int));    // 分配一个 int 数组
    MyClass* obj = (MyClass*)malloc(sizeof(MyClass)); // 分配内存，但未调用构造函数
    ```
![](Figure/cpp/new.png)

### 9. extern
**extern 是 C++ 中用于声明变量或函数在别处定义 的关键字，告诉编译器“这个符号的定义在其他地方，不要报错”。** 它是实现跨文件访问和 C/C++ 混合编程的重要机制

1. 声明变量：一般用于多文件编程，声明变量在其他文件中定义。
    ```cpp
    // file.h
    extern int globalVar;  // 声明，不是定义
    // file.cpp
    int globalVar = 10;    // 定义
    ```
2. 声明函数：一般用于多文件编程，声明函数在其他文件中定义。但是**函数其实不用声明，因为函数在编译时会自动被声明。**
    ```cpp
    // file.h
    extern void func();  // 声明，不是定义
    ```
3. extern "C"：用于声明函数在 C 语言中定义，告诉编译器“这个函数在 C 语言中定义，不要进行 C++ 的名称修饰”。(实现c++和c混合编程)
    ```cpp
    extern "C" {
        void func();
    }
    ```












## 二. C++的内存管理

## 三. C++面向对象编程

## 四. C++的STL

## 五. C++泛型编程

## 六. C++11新特性