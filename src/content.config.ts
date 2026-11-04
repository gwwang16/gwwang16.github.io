import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { file, glob } from 'astro/loaders';

const papers = defineCollection({
  loader: file('./src/content/selected-publications.json'),
  schema: z.object({
    title: z.string(),
    authors: z.array(z.string()).min(2),
    year: z.number().int(),
    journal: z.string(),
    volume: z.string().optional(),
    issue: z.string().optional(),
    pages: z.string().optional(),
    doi: z.string().startsWith('10.'),
    roles: z.array(z.enum(['first', 'corresponding'])).min(1),
    language: z.enum(['en', 'zh-CN']).default('en'),
    online: z.boolean().default(false),
    featuredOrder: z.number().int().optional(),
    verifiedOn: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
    sources: z.object({
      metadata: z.url(),
      authorship: z.url(),
      note: z.string(),
    }),
  }),
});

const research = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/research' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    papers: z.array(z.string()),
    order: z.number().int(),
  }),
});

const articles = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/articles' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    permalink: z.string().startsWith('/posts/'),
    tags: z.array(z.string()).default([]),
  }),
});

const publications = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/publications' }),
  schema: z.object({
    title: z.string(),
    excerpt: z.string(),
    date: z.coerce.date(),
    year: z.number().optional(),
    venue: z.string().optional(),
    permalink: z.string().startsWith('/publication/'),
    paperurl: z.url().optional(),
    authors: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    slug: z.string(),
    category: z.enum(['Vehicles', 'Robotics']),
    image: z.string(),
    imageAlt: z.string().optional(),
    animation: z.string().optional(),
    technologies: z.array(z.string()),
    repository: z.url().optional(),
    article: z.string().optional(),
    legacyPath: z.string(),
    featured: z.boolean().default(false),
  }),
});

const pages = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/pages' }),
  schema: z.object({ title: z.string(), permalink: z.string() }),
});

export const collections = {
  papers,
  research,
  articles,
  publications,
  projects,
  pages,
};
