import { sidebar } from "vuepress-theme-hope";

export const zhSidebar = sidebar({
  "/zh/": [
    "",
    {
      text: "学习笔记",
      icon: "folder-open",
      prefix: "notes/",
      link: "notes/",
      children: [
        {
          text: "AI infra",
          link: "ai-infra/",
        },
        {
          text: "论文浅读",
          link: "papers/",
        },
        {
          text: "CUDA",
          link: "cuda/",
        },
        {
          text: "基础知识",
          link: "basic/",
        },
      ],
    },
    {
      text: "项目实践",
      icon: "code",
      prefix: "projects/",
      link: "/zh/projects/",
      children: [
        {
          text: "GPU加速图像处理系统",
          link: "gpu-image-processing-system/",
        },
        {
          text: "Mac 如何安装 Openclaw",
          link: "Openclaw/",
        },
      ],
    },
    {
      text: "Book",
      icon: "book",
      prefix: "book/",
      link: "/zh/book/",
      children: [
        {
          text: "《延迟满足》读书随记",
          link: "delay-gratification/",
        },
      ],
    },
    {
      text: "关于我",
      icon: "profile",
      link: "/zh/about/",
    },
  ],
});
