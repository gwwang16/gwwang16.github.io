# Architecture

## Rendering and ownership

Astro generates a public academic site with static HTML, TypeScript, schema-validated content, and small progressive enhancements. A server application or client SPA is unnecessary for the research profile and bibliography. Individual interactive research demonstrations can be added later without changing this model.

```text
src/data/profile.ts                    → biography, funding, teaching, contact
src/content/selected-publications.json  → verified citation records and authorship
src/content/research/*.md               → current research and future results
src/content/projects/*.md               → early-work summary metadata
src/content/articles/*.md               → preserved historical source
src/content/publications/*.md           → preserved historical source / legacy URLs
src/lib/                               → ordering, bibliography export, old URL mapping
src/components/                        → citations, research, timeline, shared navigation
src/layouts/BaseLayout                 → SEO, person metadata, header, footer
src/pages/                             → static routes and compatibility redirects
src/assets/guangwei-wang.jpg            → current portrait, build-time optimization
public/                                → original media, historical CV, custom domain
                                       ↓
                                     dist/
                                       ↓
                                 GitHub Pages
```

The bibliography uses full author lists in data and compact reference-style rendering. Only papers with a verified first-author or corresponding-author role are admitted. Names are highlighted and verified correspondence is marked with an asterisk. Titles resolve directly through DOI. The same records generate downloadable BibTeX, so citations and export do not drift.

Each research Markdown document creates a standalone route and homepage summary. Related paper IDs connect the overview to the verified bibliography. This is the extension point for future research results: add meaningful research content without introducing an empty “coming soon” section.

## Routes

| Route                                          | Behavior                                                                                                                      |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `/`                                            | Academic profile, research, six representative papers, current projects, background, teaching, early-work collection, contact |
| `/research/<id>/`                              | Current research overview and related references                                                                              |
| `/publications/`                               | Selected first-author / corresponding-author papers, newest first                                                             |
| `/publications.bib`                            | Static BibTeX export of the same records                                                                                      |
| `/projects/`                                   | Eight concise student-work summaries with original code/demo links                                                            |
| `/projects/<slug>/`, `/posts/<original-path>/` | Redirect to the corresponding `/projects/#id`                                                                                 |
| `/publication/<original-slug>/`                | Redirect to the corresponding updated citation                                                                                |
| `/notes/`                                      | Redirect to the early-work collection                                                                                         |
| `/terms/`                                      | Original privacy policy                                                                                                       |
| `/sitemap/`, `/sitemap-index.xml`              | Human and crawler indexes                                                                                                     |
| `/404.html`                                    | Recovery page for GitHub Pages                                                                                                |

Old about, portfolio, and taxonomy routes remain compatible. GitHub Pages has no arbitrary server redirect rules, so aliases use immediate HTML redirects with a visible fallback link, canonical metadata, and noindex. They are excluded from XML sitemaps. Physical `.html` aliases remain in `public` to avoid unwanted nested route output.

## Content and evidence

The immutable migration manifest covers nineteen original bodies: six articles, seven projects, five publications, and privacy. These remain in source. The user requested condensed student work, so original tutorials are not rendered on the public site. The home-service robot article contributes the eighth early-work item. Two historical publication drafts remain private; they are not silently assumed published.

Current profile facts, funding, courses, mentoring, portrait, public email, and university URL come from the faculty profile updated in April 2026. Recent publications were discovered through the correct Guizhou University ResearchGate profile and checked against publishers / Crossref. Google Scholar’s existing link is retained, but its direct page returned HTTP 429 during verification. See [SOURCES.md](SOURCES.md) and each paper’s source fields.

Issue metadata takes precedence over initial online dates. For example, the adjustable microgripper is cited as Micromachines 2024 despite its December 2023 online release, and the robust nanopositioning paper is cited as Journal of Vibration and Control 2023 despite appearing online in 2022.

The theme’s sample teaching/talks/pages, Liquid/Ruby toolchain, old jQuery bundle, and committed Sass caches are retired. Real original image/CV paths, analytics ID, domain, and MIT license are preserved. The old CV is kept as a historical asset, not advertised as a current profile document.

## Accessibility and delivery

The native mobile menu works without JavaScript. A small script adds Escape, outside-click, and link closing. Static content never depends on client API calls or JavaScript for visibility. Images reserve dimensions; the current portrait is generated as responsive WebP and a compressed social preview.

The shared Markdown renderer retains build-time code highlighting, math/MathML, accessible table scrolling, and opt-in animation support for future technical research content. Old Markdown bodies need no destructive rewrite.

`npm run validate` runs formatting, Astro/TypeScript diagnostics, the production build, and content/design checks. Real Chromium checks exercise routes, keyboard use, no-JavaScript reading, bibliography export, narrow-screen layout, and automated accessibility. PR builds cannot deploy. Production deployment targets `master` and the `github-pages` environment.

The current production Pages source is the legacy branch/Jekyll mode. Deployment requires selecting GitHub Actions as the Pages source and merging the reviewed branch. This task does not change Pages settings or DNS. Rollback is a revert plus restoration of the previous Pages source if returning to Jekyll.

Chromium and automated WCAG checks are the tested scope; Safari/Firefox and screen-reader testing remain additional release coverage.
