# Guangwei Wang · Academic website

Personal academic website of Guangwei Wang (王广玮), Associate Professor at Guizhou University. Built with Astro and TypeScript, hosted at [www.guangwei.wang](https://www.guangwei.wang).

## Local development

Requires Node.js 22.12 or later and npm.

```sh
npm ci
npm run dev
```

Preview at [localhost:4321](http://localhost:4321). Run `npm run validate` before publishing to check formatting, types, the build, and content links.

## Content updates

| Content                                                     | Location                                         |
| ----------------------------------------------------------- | ------------------------------------------------ |
| Profile, research projects, teaching, recruitment, and book | `src/data/profile.ts`                            |
| Papers, author roles, DOIs, and sources                     | `src/content/selected-publications.json`         |
| Research overviews and related papers                       | `src/content/research/*.md`                      |
| Earlier project summaries                                   | `src/content/projects/*.md`                      |
| Images and portrait                                         | `public/images/`, `src/assets/guangwei-wang.jpg` |

Set a paper's `featuredOrder` to include it on the homepage. The publication list and BibTeX export are generated from the same records.

See [sources](docs/SOURCES.md), [architecture](docs/ARCHITECTURE.md), and [design](DESIGN.md) for details.

## GitHub Pages deployment

Select **GitHub Actions** under **Settings → Pages → Build and deployment**. The [Pages workflow](.github/workflows/pages.yml) validates and deploys `dist` on pushes to `master`; pull requests run checks only.

Original theme: [academicpages](https://academicpages.github.io) / [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/), © 2016 Michael Rose. See [LICENSE](LICENSE).
