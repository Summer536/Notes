import { sidebar } from "vuepress-theme-hope";

export const zhSidebar = sidebar({
  "/zh/": [
    "",
    {
      text: "学习笔记",
      icon: "folder-open",
      link: "/zh/notes/",
    },
    {
      text: "论文浅读",
      icon: "book",
      link: "/zh/interview/",
    },
    {
      text: "项目实践",
      icon: "code",
      link: "/zh/projects/",
    },
  ],
});
