import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { accessibleMarkdown } from './scripts/accessible-markdown.mjs';

export default defineConfig({
  site: 'https://www.guangwei.wang',
  output: 'static',
  trailingSlash: 'always',
  integrations: [
    sitemap({
      filter: (page) => {
        const path = new URL(page).pathname;
        if (
          path.startsWith('/posts/') ||
          path.startsWith('/publication/') ||
          path === '/notes/' ||
          /^\/projects\/[^/]+\//.test(path)
        )
          return false;
        return ![
          '/404.html',
          '/about/',
          '/about.html',
          '/blog/',
          '/portfolio/',
          '/categories/',
          '/tags/',
          '/page-archive/',
          '/collection-archive/',
          '/publications.html',
        ].some((alias) => path.startsWith(alias));
      },
    }),
  ],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, accessibleMarkdown],
    }),
    shikiConfig: { theme: 'github-light', wrap: true },
  },
});
