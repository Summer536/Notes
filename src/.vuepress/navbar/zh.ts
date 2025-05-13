import { navbar } from "vuepress-theme-hope";

export const zhNavbar = navbar([
  "/zh/",
  {
    text: "博文",
    icon: "pen-to-square",
    prefix: "/zh/posts/",
    children: [
      { text: "CUDA技术栈", icon: "pen-to-square", link: "cuda-tech-stack" }
    ],
  },
]);
