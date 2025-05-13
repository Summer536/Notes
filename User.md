# 博客使用手册

## 一、创建新文章

### 1. 创建Markdown文件

在 `src/zh/posts/` 目录下创建一个新的 `.md` 文件，文件名建议使用英文，例如 `new-article.md`。

### 2. 添加文章元数据

每篇文章开头需要添加如下格式的元数据：

```
---
title: 文章标题
date: 2023-10-30
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