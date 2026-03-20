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
      text: "Book",
      icon: "code",
      link: "/zh/book/",
    },
    {
      text: "关于我",
      icon: "profile",
      link: "/zh/about/",
    },
  ],
});
