import { sidebar } from "vuepress-theme-hope";

export const enSidebar = sidebar({
  "/": [
    "",
    {
      text: "Notes",
      icon: "folder-open",
      prefix: "notes/",
      link: "notes/",
      children: [
        {
          text: "AI infra",
          link: "ai-infra/",
        },
        {
          text: "Paper Reading",
          link: "papers/",
        },
        {
          text: "CUDA",
          link: "cuda/",
        },
        {
          text: "Fundamentals",
          link: "basic/",
        },
      ],
    },
    {
      text: "Projects",
      icon: "code",
      link: "/projects/",
    },
    {
      text: "Book",
      icon: "book",
      link: "/book/",
    },
    {
      text: "About",
      icon: "profile",
      link: "/about/",
    },
  ],
});
