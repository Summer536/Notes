import { navbar } from "vuepress-theme-hope";

export const enNavbar = navbar([
  "/en/",
  {
    text: "Notes",
    icon: "folder-open",
    link: "/en/notes/",
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
]);
