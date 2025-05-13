# 博客使用手册

## 一、创建新文章

### 1. 创建Markdown文件

在 `src/zh/posts/` 目录下创建一个新的 `.md` 文件，文件名建议使用英文，例如 `new-article.md`。

### 2. 添加文章元数据

每篇文章开头需要添加如下格式的元数据：

```
---
title: 文章标题
date: 2025-05-10  # 日期越新，排序越靠前
category:
  - 分类1
  - 分类2
tag:
  - 标签1
  - 标签2
cover: /assets/images/cover1.jpg
isOriginal: true
---
```

特别说明：
- `date`: 文章发布日期，用于排序，日期越新的文章排序越靠前
- `cover`: 文章封面图片，可以从 `/assets/images/` 目录中选择
- `isOriginal`: 是否为原创文章
- 添加 `<!-- more -->` 标签可以控制文章在列表中的摘要显示长度

### 3. 编写文章内容

使用Markdown语法编写文章内容。支持标题、列表、链接、图片、代码块等格式。

示例文章内容：

```
# 文章标题

## 简介部分

这是简介内容...

<!-- more -->

## 第二部分

这是更多内容...

### 代码示例

​```python
def hello():
    print("Hello world")
​```
```

### 4. 将文章归类到不同分类

网站有以下几个主要分类：
- **学习笔记** - 存放在 `src/zh/notes/` 目录
- **八股文总结** - 存放在 `src/zh/interview/` 目录
- **项目实践** - 存放在 `src/zh/projects/` 目录

#### 方法1: 在分类目录中创建文章

直接在对应目录中创建Markdown文件。例如，要添加一篇学习笔记：

```
src/zh/notes/your-new-note.md
```

#### 方法2: 在posts目录创建并链接到分类

1. 在 `src/zh/posts/` 目录创建文章
2. 在对应分类的README.md中添加链接

例如，要在学习笔记中添加一个链接：

```markdown
// 在 src/zh/notes/README.md 中添加
- [新文章标题](/zh/posts/your-new-article.html) - 文章简介
```

#### 方法3: 使用category元数据自动分类

在文章的元数据中设置适当的category，VuePress会自动将其归类：

```yaml
---
title: 你的文章标题
date: 2025-05-15
category:
  - 学习笔记  # 这将自动分类到"学习笔记"类别
---
```

## 二、本地预览

### 1. 启动开发服务器

在终端中运行：

```
npm run docs:dev
```

### 2. 访问本地网站

打开浏览器，访问 http://localhost:8080/ 即可预览网站效果。

## 三、推送至GitHub

### 1. 首次设置（已完成）

Git仓库已初始化并关联到GitHub远程仓库：

```
# 远程仓库已设置为
https://github.com/Summer536/Notes.git
```

### 2. 推送更新内容

每次修改完成后，执行以下命令推送至GitHub：

```
# 添加所有更改
git add .

# 提交更改（引号内填写提交说明）
git commit -m "添加了新文章：文章标题"

# 推送至GitHub
git push
```

### 3. 解决可能的问题

如果推送遇到问题，可能需要先拉取远程代码：

```
git pull --rebase
```

然后再次尝试推送：

```
git push
```

## 四、查看GitHub Pages网站

### 1. 自动部署说明

本项目已配置好GitHub Actions自动部署工作流（`.github/workflows/deploy-docs.yml`）。当您将代码推送到GitHub的main分支时，会自动触发以下操作：

1. 检出代码
2. 设置Node.js环境（使用v18）
3. 安装依赖
4. 构建文档
5. 将构建结果部署到gh-pages分支

您不需要手动运行任何部署命令，GitHub会自动处理所有部署步骤。

### 2. 启用GitHub Pages

首次部署需在GitHub仓库设置中启用GitHub Pages：

1. 打开您的GitHub仓库页面：https://github.com/Summer536/Notes
2. 点击"Settings"（设置）选项卡
3. 滚动到左侧菜单的"Pages"部分
4. Source选择"Deploy from a branch"
5. Branch选择"gh-pages" / "/(root)"
6. 点击"Save"按钮

### 3. 访问您的网站

设置完成后，您可以通过以下地址访问您的博客：

```
https://summer536.github.io/Notes/
```

推送代码后，部署过程通常需要1-3分钟完成。

### 4. 查看部署状态

您可以在GitHub仓库的"Actions"选项卡中查看部署进度和状态：
https://github.com/Summer536/Notes/actions

如果部署失败，可以在此查看错误日志。

## 五、网站定制与优化

### 1. 修改首页背景图

首页背景图位于 `src/zh/README.md` 文件中配置：

```yaml
---
home: true
layout: Blog
icon: house
title: 博客主页
heroImage: /logo.svg
bgImage: /assets/images/cover1.jpg  # 这里修改背景图
heroText: GYQ的技术博客
heroFullScreen: true
tagline: 记录学习历程，分享技术心得
---
```

可用的背景图有：
- `/assets/images/cover1.jpg`
- `/assets/images/cover2.jpg`
- `/assets/images/cover3.jpg`

也可以将自己的图片放在 `src/.vuepress/public/assets/images/` 目录下。

### 2. 修改站点名称和描述

站点名称和描述在 `src/.vuepress/config.ts` 文件中配置：

```typescript
locales: {
  "/": {
    lang: "en-US",
    title: "GYQ's Blog",  // 英文网站标题
    description: "Personal technical blog and learning notes",  // 英文网站描述
  },
  "/zh/": {
    lang: "zh-CN",
    title: "GYQ的博客",  // 中文网站标题
    description: "个人技术博客与学习笔记",  // 中文网站描述
  },
}
```

### 3. 自定义导航栏和侧边栏

导航栏配置文件: `src/.vuepress/navbar/zh.ts`
侧边栏配置文件: `src/.vuepress/sidebar/zh.ts`

### 4. 修改Logo和页脚

- **Logo**: 替换 `src/.vuepress/public/logo.svg` 文件
- **页脚**: 在主页配置中修改 `footer` 属性

### 5. 添加和使用图片

#### 5.1 图片目录结构

图片文件应存放在以下位置：
- 站点公共图片: `src/.vuepress/public/images/`
- 文章封面图片: `src/.vuepress/public/assets/images/`
- 文章内部图片: 可以放在上述任一目录

#### 5.2 在文章中引用图片

在Markdown文件中引用图片的方式：

```markdown
<!-- 方式1：引用公共images目录下的图片 -->
![图片描述](/Notes/images/example.jpg)

<!-- 方式2：引用assets/images目录下的图片 -->
![图片描述](/Notes/assets/images/cover1.jpg)

<!-- 方式3：引用网络图片 -->
![图片描述](https://example.com/image.jpg)
```

#### 5.3 添加新图片

1. 将您的图片放入 `src/.vuepress/public/images/` 目录
2. 在文章中使用相对路径引用，例如 `/Notes/images/your-image.jpg`
3. 图片文件名建议使用英文和连字符，避免使用空格和特殊字符

### 6. 修改首页导航链接

首页导航链接在 `src/zh/README.md` 文件的 `projects` 部分定义：

```yaml
projects:
  - icon: book
    name: 八股文总结
    desc: 系统性整理的面试知识点
    link: /interview/

  - icon: folder-open
    name: 学习笔记
    desc: 日常技术学习笔记
    link: /notes/

  # 其他链接...
```

#### 6.1 添加新链接

按照现有格式添加新的项目块：

```yaml
  - icon: heart  # 图标名称，基于FontAwesome图标
    name: 新链接名称  # 显示的链接名称
    desc: 新链接描述信息  # 链接描述
    link: /zh/your-link-path/  # 链接路径
```

可用的图标可以在 [Font Awesome 图标库](https://fontawesome.com/icons) 中查找。

#### 6.2 删除链接

删除对应的整个项目块即可。

#### 6.3 修改链接

修改现有项目块中的属性值即可。

注意：`link` 属性应指向有效的页面路径，例如：
- `/zh/posts/` - 指向中文博客文章列表
- `/zh/about/` - 指向"关于我"页面

## 六、常见问题

### 1. 图片引用

图片文件应放在 `src/.vuepress/public/images/` 目录下，然后在Markdown中这样引用：

```
![图片描述](/Notes/images/图片名.jpg)
```

### 2. 本地预览与线上不一致

确保您已正确设置了 `config.ts` 中的 `base` 配置项为 `/Notes/`。

### 3. 如何删除文章

直接删除对应的Markdown文件，然后推送到GitHub即可。

### 4. 部署失败常见原因

- Node.js版本不兼容：工作流配置使用Node.js 18
- 内存不足：构建大型文档可能需要增加内存限制
- 语法错误：Markdown或配置文件中的语法错误可能导致构建失败 