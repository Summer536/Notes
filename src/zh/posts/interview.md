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

### 12. 模版函数有哪些编译方式？以下模板代码编译能否通过？怎么样修改让它通过？
一体化编译会增加编译时间，分离式编译会多次重复显式实例化，且多个cpp可能实例化同一模板造成维护困难，后者可能造成可执行文件膨胀。

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








；前者可能出现，多次重复显式实例化，且多个cpp可能实例化同一模板造成维护困难，后者可能造成可执行文件膨胀










## 二. 操作系统


## 三. AI-HPC基础


## 四. X86CPU体系结构

## 五. GPU体系结构与cuda编程

## 六. 大模型量化

## 七. 大模型推理

## 八. Leetcode