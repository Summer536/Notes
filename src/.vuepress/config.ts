import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/Notes/",

  head: [
    ["link", { rel: "icon", href: "/Notes/favicon.ico" }],
    ["link", { rel: "shortcut icon", href: "/Notes/favicon.ico" }],
    ["link", { rel: "apple-touch-icon", href: "/Notes/favicon.ico" }],
  ],

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
