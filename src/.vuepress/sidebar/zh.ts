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
      link: "/zh/projects/",
    },
    {
      text: "Book",
      icon: "book",
      link: "/zh/book/",
    },
    {
      text: "关于我",
      icon: "profile",
      link: "/zh/about/",
    },
  ],
});
