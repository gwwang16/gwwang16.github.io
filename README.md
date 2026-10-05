# Guangwei Wang · Academic website

Academic website of Guangwei Wang (王广玮), Associate Professor at Guizhou University. Hosted at **https://www.guangwei.wang** from **gwwang16/gwwang16.github.io**.

Built with **Astro 7 and TypeScript**. Research, publications, and profile content are generated as static HTML. Reading and navigation work without a client framework or application server.

## Local development

Use Node.js 24 (minimum 22.12) and npm.

```sh
npm ci
npm run dev
```

Open `http://localhost:4321`. To inspect a production build:

```sh
npm run build
npm run preview
```

## Updating academic content

- `src/data/profile.ts`: current biography, appointments, funded projects, teaching, recruitment, public contact, and academic links.
- `src/content/selected-publications.json`: selected first-author / corresponding-author papers. Each record includes full authors, final citation metadata, DOI, verified author roles, and source URLs. Never infer correspondence from author order.
- `src/content/research/*.md`: research overviews and future results. Add a Markdown document with `title`, `summary`, `icon`, `topics`, related `papers` IDs, and `order`; it gets its own research page and homepage entry.
- `src/content/projects/*.md`: metadata for concise early-work summaries. Their original bodies remain archived in source; they are not rendered as long tutorials.
- `src/content/articles/*.md` and `src/content/publications/*.md`: original source archive and legacy URL metadata. Existing links resolve to the relevant early-work summary or updated citation.
- `src/content/pages/terms.md`: original privacy policy.
- `public/images/`, `public/files/`: preserved public images and the historical CV. The current portrait is `src/assets/guangwei-wang.jpg` and is optimized at build time.

For a new paper, verify its role against a publisher author note or the university profile, add complete metadata and sources, and reference its ID from the relevant research overview. Set `featuredOrder` only for representative papers to show on the homepage. New entries automatically join the bibliography and downloadable `/publications.bib`.

English remains the main language, with the Chinese name, project titles, course names, and Chinese-journal citations retained. [docs/SOURCES.md](docs/SOURCES.md) records provenance and citation-year corrections.

Schemas in `src/content.config.ts` validate content. Layout/components live in `src/layouts`, `src/components`, and `src/pages`. Visual tokens are owned by `src/styles/global.css` and documented in [DESIGN.md](DESIGN.md).

## Verification

```sh
npm run validate
npx playwright install chromium --only-shell
npm run test:browser
```

Checks cover formatting, TypeScript, static builds, original-body integrity, draft exclusion, legacy redirects, bibliography metadata/export, internal links/assets, research references, academic timeline, and design-token drift. Real Chromium tests cover keyboard navigation, native mobile menus, no-JavaScript use, citation links, 320/768/1440px layouts, and automated WCAG checks. Test requests to the existing Google Analytics property are suppressed.

If the home directory is unwritable, set `ASTRO_TELEMETRY_DISABLED=1` and use a writable npm cache and `PLAYWRIGHT_BROWSERS_PATH`.

## GitHub Pages deployment

The existing production site uses **Deploy from a branch → master → /** and GitHub’s Jekyll builder. Astro uses the checked-in Actions workflow.

When publishing the reviewed branch:

1. Open **Settings → Pages → Build and deployment** and select **GitHub Actions** as the source.
2. Merge the reviewed pull request into `master`.
3. The workflow validates the site and uploads `dist` to Pages. Deployment runs only from `master`; pull requests build and test without deploying.

`public/CNAME` preserves `www.guangwei.wang`. Existing DNS remains applicable. The refactor branch does not change production Pages settings.

## Architecture and migration

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes rendering, routing, and maintenance. [docs/content-migration.json](docs/content-migration.json) records all 19 original content documents and their body hashes. Their full text remains available in repository history and the source archive, while the website presents condensed student work and verified academic citations.

The original theme was [academicpages](https://academicpages.github.io) / [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/), © 2016 Michael Rose, MIT licensed. The original [LICENSE](LICENSE) is retained.
