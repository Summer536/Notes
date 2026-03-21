import { sidebar } from "vuepress-theme-hope";

export const enSidebar = sidebar({
  "/en/": [
    "",
    {
      text: "Notes",
      icon: "folder-open",
      prefix: "notes/",
      link: "/en/notes/",
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
      link: "/en/projects/",
    },
    {
      text: "Book",
      icon: "book",
      link: "/en/book/",
    },
    {
      text: "About",
      icon: "profile",
      link: "/en/about/",
    },
  ],
});
