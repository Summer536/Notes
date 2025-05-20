import { navbar } from "vuepress-theme-hope";

export const zhNavbar = navbar([
  "/zh/",
  {
    text: "学习笔记",
    icon: "folder-open",
    link: "/zh/notes/",
  },
  {
    text: "项目实践",
    icon: "code",
    link: "/zh/projects/",
  },
  // {
  //   text: "博文",
  //   icon: "pen-to-square",
  //   prefix: "/zh/posts/",
  //   children: [
  //     { text: "CUDA技术栈", icon: "pen-to-square", link: "cuda-tech-stack" },
  //     { text: "Flashattention", icon: "pen-to-square", link: "flashattention" }
  //   ],
  // },
]);
