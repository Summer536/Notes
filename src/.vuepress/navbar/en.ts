import { navbar } from "vuepress-theme-hope";

export const enNavbar = navbar([
  "/",
  {
    text: "Notes",
    icon: "folder-open",
    link: "/notes/",
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
]);
