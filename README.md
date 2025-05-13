# GYQ的技术博客

## 更新网站流程

1. **修改内容**：编辑`src`目录下的Markdown文件或配置文件

2. **本地预览**：
   ```bash
   npm run docs:dev
   ```
   本地访问 http://localhost:8080/Notes/ 预览效果

3. **构建网站**：
   ```bash
   npm run docs:build
   ```
   这会在`.vuepress/dist`目录生成静态文件

4. **提交更改**：
   ```bash
   git add .
   git commit -m "更新说明"
   git push
   ```

5. **等待部署**：
   - GitHub Pages会自动部署更新（通常需要几分钟时间）
   - 如果网页没有更新，可能是浏览器缓存问题，尝试强制刷新（Ctrl+F5或Cmd+Shift+R）

## 常见问题

- **本地预览正常但网站未更新**：检查是否完成了构建和推送步骤
- **浏览器缓存**：使用强制刷新或清除浏览器缓存
- **修改未生效**：确保修改了正确的文件，并且frontmatter格式正确

## 文件结构

- `src/`：网站源代码
  - `zh/`：中文内容
  - `.vuepress/`：配置和主题文件
  - `posts/`：博客文章