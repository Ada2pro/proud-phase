# Proud Phase 🔐

![Typescript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![Astro](https://img.shields.io/badge/Astro-FF5D01?style=for-the-badge&logo=astro&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

这是我的个人技术博客，专注于**全同态加密（FHE）**和**密码学**相关的深度技术文章。博客基于 AstroPaper 主题构建，提供快速、响应式且 SEO 友好的阅读体验。

## 📚 博客内容

本博客主要记录我在学习和研究全同态加密过程中的思考与总结，包括：

- **FHE 论文精读**：深入解析前沿论文，如 CKKS、BFV、TFHE 等方案
- **算法实现**：同态加密算法的实现细节与优化技巧
- **数学原理**：密码学背后的数学基础与证明
- **工程实践**：RNS-CKKS、Bootstrapping 等技术的工程化实现
- **学习笔记**：与 AI 对话的学习记录，深入浅出地理解复杂概念

## ✨ 博客特色

- [x] **数学公式支持**：集成 KaTeX，完美渲染 LaTeX 数学公式
- [x] **深度技术文章**：详细的论文解读和算法分析
- [x] **问答式学习**：通过对话形式记录学习过程，便于理解
- [x] **代码示例**：提供实际的代码实现和优化技巧
- [x] **响应式设计**：在各种设备上都有良好的阅读体验
- [x] **暗色模式**：保护眼睛的深色主题
- [x] **快速搜索**：模糊搜索功能，快速定位内容
- [x] **SEO 优化**：便于搜索引擎收录和分享
- [x] **标签分类**：按主题组织文章（FHE、密码学、算法等）
- [x] **RSS 订阅**：支持 RSS feed 订阅更新

## 🎯 最新文章

- **阅读论文《Homomorphic Multiple Precision Multiplication for CKKS and Reduced Modulus Consumption》**
  - 深入解析 CKKS 方案的多精度乘法优化
  - 详细讲解 Mult² 算法的原理与实现
  - 探讨模数消耗减半的数学原理

## 🚀 项目结构

博客的目录结构如下：

```bash
/
├── public/
│   ├── blog/              # 博客文章的图片资源
│   │   └── CKKS-DR/       # 各篇文章的图片文件夹
│   └── favicon.svg
│   └── toggle-theme.js
├── src/
│   ├── components/        # React/Astro 组件
│   ├── data/
│   │   └── blog/          # 📝 博客文章（Markdown 格式）
│   │       └── CKKS-DR.md # 论文阅读笔记
│   ├── layouts/           # 页面布局模板
│   ├── pages/             # 路由页面
│   ├── styles/            # 全局样式
│   └── config.ts          # 博客配置
├── astro.config.ts        # Astro 配置（含 KaTeX 支持）
└── package.json
```

**重要目录说明**：
- `src/data/blog/`：所有博客文章的 Markdown 文件
- `public/blog/`：文章中使用的图片和静态资源
- `astro.config.ts`：已配置 remark-math 和 rehype-katex 用于数学公式渲染

## 📝 如何添加新文章

1. 在 `src/data/blog/` 目录下创建新的 Markdown 文件
2. 添加 frontmatter 元数据：

```markdown
---
title: "文章标题"
pubDatetime: 2025-12-23T10:00:00Z
description: "文章简介"
tags:
  - FHE
  - CKKS
featured: true
draft: false
timezone: "Asia/Shanghai"
---

文章内容...
```

3. 如果文章包含图片，将图片放在 `public/blog/文章名/` 目录下
4. 在 Markdown 中使用相对路径引用图片：`![描述](/blog/文章名/图片.png)`
5. 数学公式使用 LaTeX 语法：
   - 行内公式：`$E = mc^2$`
   - 块级公式：`$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$`

## 💻 技术栈

**主框架** - [Astro](https://astro.build/) - 现代化的静态站点生成器  
**类型检查** - [TypeScript](https://www.typescriptlang.org/) - 类型安全  
**样式** - [TailwindCSS](https://tailwindcss.com/) - 实用优先的 CSS 框架  
**数学公式** - [KaTeX](https://katex.org/) - 快速的数学公式渲染  
**Markdown 处理** - [remark-math](https://github.com/remarkjs/remark-math) + [rehype-katex](https://github.com/remarkjs/remark-math/tree/main/packages/rehype-katex)  
**搜索功能** - [Pagefind](https://pagefind.app/) - 静态站点搜索  
**图标** - [Tabler Icons](https://tabler-icons.io/)  
**代码格式化** - [Prettier](https://prettier.io/)  
**代码检查** - [ESLint](https://eslint.org)  
**部署** - [GitHub Pages](https://pages.github.com/) / [Vercel](https://vercel.com/) / [Netlify](https://www.netlify.com/)

## 🚀 本地运行

### 克隆项目

```bash
git clone https://github.com/你的用户名/proud-phase.git
cd proud-phase
```

### 安装依赖

```bash
pnpm install
```

### 启动开发服务器

```bash
pnpm run dev
```

然后在浏览器中访问 `http://localhost:4321`

### 构建生产版本

```bash
pnpm run build
```

构建完成后，静态文件将生成在 `dist/` 目录中。

### Docker 部署（可选）

如果你安装了 Docker，也可以使用 Docker 运行：

```bash
# 构建 Docker 镜像
docker build -t proud-phase .

# 运行容器
docker run -p 4321:80 proud-phase
```

## ⚙️ 配置

### 博客基本信息

在 `src/config.ts` 中修改博客的基本信息：

```typescript
export const SITE = {
  website: "https://你的域名.com/",
  author: "你的名字",
  desc: "专注于全同态加密和密码学的技术博客",
  title: "Proud Phase",
  // ... 其他配置
};
```

### Google 站点验证（可选）

在 `.env` 文件中添加：

```bash
PUBLIC_GOOGLE_SITE_VERIFICATION=your-verification-code
```

## 🧞 常用命令

所有命令都在项目根目录下的终端中运行：

| 命令                     | 说明                                                                     |
| :----------------------- | :----------------------------------------------------------------------- |
| `pnpm install`           | 安装依赖                                                                 |
| `pnpm run dev`           | 启动本地开发服务器，访问 `localhost:4321`                               |
| `pnpm run build`         | 构建生产版本到 `./dist/` 目录                                           |
| `pnpm run preview`       | 在本地预览构建结果                                                       |
| `pnpm run format:check`  | 使用 Prettier 检查代码格式                                               |
| `pnpm run format`        | 使用 Prettier 格式化代码                                                 |
| `pnpm run sync`          | 为所有 Astro 模块生成 TypeScript 类型                                    |
| `pnpm run lint`          | 使用 ESLint 检查代码                                                     |

### Docker 命令（可选）

| 命令                                  | 说明                                                   |
| :------------------------------------ | :----------------------------------------------------- |
| `docker compose up -d`                | 使用 Docker Compose 运行博客                           |
| `docker compose run app pnpm install` | 在 Docker 容器中运行命令                               |
| `docker build -t proud-phase .`       | 构建 Docker 镜像                                       |
| `docker run -p 4321:80 proud-phase`   | 运行 Docker 容器，访问 `http://localhost:4321`         |

## 🤝 贡献

欢迎提出建议和反馈！如果你发现了 bug 或有新功能建议，请：

1. 提交 Issue
2. 发起 Pull Request
3. 或通过邮件联系我

## 📄 许可证

本项目基于 MIT License 开源。

## 🙏 致谢

- 博客主题基于 [AstroPaper](https://github.com/satnaing/astro-paper) 构建
- 感谢所有开源项目的贡献者

---

**专注于全同态加密，探索密码学的奥秘** 🔐
