import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/Notes/",

  locales: {
    "/": {
      lang: "en-US",
      title: "GYQ's Blog",
      description: "Personal technical blog and learning notes",
    },
    "/zh/": {
      lang: "zh-CN",
      title: "GYQ的博客",
      description: "个人技术博客与学习笔记",
    },
  },

  theme,

  // Enable it with pwa
  // shouldPrefetch: false,
});
