---
version: alpha
name: Guangwei Wang
description: A compact engineering faculty homepage with reference-style publications and restrained blue accents.
colors:
  primary: '#245a82'
  primary-hover: '#183e5b'
  ink: '#242b33'
  muted: '#57606a'
  background: '#ffffff'
  paper: '#f4f6f8'
  soft: '#fafbfc'
  border: '#d9dfe5'
  accent: '#294d6d'
  focus: '#0866c5'
typography:
  display:
    fontFamily: "Arial, Helvetica, 'Noto Sans SC', sans-serif"
  body:
    fontFamily: "Arial, Helvetica, 'Noto Sans SC', sans-serif"
    fontSize: '16px'
    lineHeight: '1.6'
  mono:
    fontFamily: "'SFMono-Regular', Consolas, monospace"
rounded:
  control: '2px'
  panel: '0px'
spacing:
  page-width: '960px'
  section-space: '28px'
components:
  document:
    backgroundColor: '{colors.background}'
    textColor: '{colors.ink}'
  heading:
    textColor: '{colors.accent}'
  link:
    textColor: '{colors.primary}'
  link-hover:
    textColor: '{colors.primary-hover}'
  secondary-text:
    textColor: '{colors.muted}'
  divider:
    backgroundColor: '{colors.border}'
  technical-surface:
    backgroundColor: '{colors.paper}'
  inline-code:
    backgroundColor: '{colors.soft}'
  portrait:
    rounded: '{rounded.panel}'
  animation-control:
    height: '44px'
    rounded: '{rounded.control}'
    backgroundColor: '{colors.paper}'
    textColor: '{colors.primary}'
  focus-ring:
    backgroundColor: '{colors.focus}'
---

# Guangwei Wang Design System

## Overview

The accepted direction is a conventional engineering faculty homepage: a compact, single-column academic document. The user requests “排版简洁为主，稍微有点装饰色保证格调.” Use a white page, modest headings, clear reference lists, and a small faculty portrait. Deep blue headings and links, with thin grey-blue rules, provide the decorative color.

The primary readers are research collaborators and prospective postgraduate students. Prioritize current research, verified first-author / corresponding-author papers, funded projects, and academic background. Student-era ROS and autonomous-driving work belongs in a concise supporting collection. Section names describe their contents directly: Research interests, Selected publications, Research projects, Education and experience, Teaching, Earlier projects, and Contact.

English is the existing site language. Preserve the Chinese name, original funded-project titles, course names, and Chinese-journal citations with appropriate language attributes. Facts and author roles come from the university and publisher records in `docs/SOURCES.md`. Avoid promotional slogans, publication-count displays, empty future-result placeholders, and commercial landing-page conventions.

Token ownership uses Model B: `src/styles/global.css` is the canonical runtime source. This document mirrors accepted values; `tests/design.test.mjs` checks the mapping. Shared layouts and citation components keep every public page consistent.

## Colors

White carries all main content. The name and section headings use the dark blue accent; links use a slightly brighter blue with visible underlines. Grey-blue rules separate sections, navigation, and the footer. Charcoal body text and muted secondary metadata support long-form reading. Pale surfaces are reserved for technical tables, code, and animation controls, rather than alternating homepage bands.

`colors.<name>` maps to `--color-<name>`. Scrollbar tokens define thumb, track, hover, and active colors with standards-based properties and WebKit fallbacks. Forced colors use system colors, and focus has a distinct blue outline.

## Typography

Use the native Arial / Helvetica stack with script-capable fallbacks. No web fonts are required. Body text and references share 16px / 1.6. The homepage name is 32px, other page titles are 28px, section headings are 20px, and subsections are 16px. The name reduces to 26px on phones; bibliography text remains 16px.

`typography.display`, `.body`, and `.mono` map to their font custom properties. Avoid exaggerated title sizes, decorative punctuation, uppercase section labels, and widely spaced lettering. Metadata uses 13–14px. References wrap naturally, abbreviate author names only when rendered, bold the researcher’s name, and mark verified correspondence with an asterisk and visible legend. Full authors remain in data and BibTeX export.

## Layout

Use a centered 960px document with a 24px minimum desktop margin and 16px phone margin. Section spacing is 28px. The body is a single column; small date columns organize funding and appointments on desktop and stack on phones. Heading rules create structure without cards or colored panels.

The profile combines name, position, institution, email, academic links, and a short biography. A 120px portrait sits beside the identity block, reducing to 100px or 80px on narrow screens. Preserve its aspect ratio and use optimized responsive assets. Navigation is a plain text row without sticky behavior or a monogram. At 800px it becomes a native disclosure that expands in document flow.

The homepage sequence is identity → research interests → representative references → funded projects → education and experience → teaching → an earlier-project collection link → contact. The separate bibliography contains all selected records. Earlier-project rows use small static thumbnails, short descriptions, and source/demo links.

`spacing.page-width` and `.section-space` map to their runtime custom properties. Document scrolling owns the page; code, tables, and long formulas may scroll locally. Hash-linked references and summaries remain visible.

## Elevation & Depth

Use thin rules and spacing. No shadows, large colored bands, floating navigation, pill tags, or promotional action panels. A compact text footer closes the document. Content remains visible without client requests, animation, or loading placeholders.

## Shapes

The portrait is square-cornered. The 2px control radius applies only to technical animation controls and scrollbar styling. These map from `rounded.panel` / `.control` to their runtime tokens. Academic navigation and profile links are ordinary text links.

## Components

Research interests are plain list entries with a short summary and link to a Markdown overview. Publication rows share one component across the homepage, bibliography, and research pages. The same records generate BibTeX. Author roles require evidence; last authorship alone is insufficient. Funded projects show dates, titles, funders, and original Chinese names without unsupported role claims.

The academic appointment list preserves the original dates and overlapping positions. Courses and student achievements use ordinary bullet lists. The original book title is retained. Contact is a text section with recruitment information and office address, with email at the profile.

Earlier-project summaries are readable without JavaScript. Old tutorials and portfolio URLs redirect to matching summary anchors; original bodies remain in source. Historical paper URLs redirect to updated references. Draft records do not generate public routes. The historical CV remains an asset; the current university page is the visible institutional reference.

The Markdown renderer supports highlighted code, math/MathML, accessible tables, static posters, and opt-in animation controls for future research content. Animation failure returns to a poster, and reduced-motion changes pause playback.

The native menu supports Escape, outside-click closing, and navigation closing without claiming modal behavior. Supplementary source/demo icons are hidden from assistive technology. The 404 page offers plain home and earlier-project recovery links. Print styles remove navigation and preserve readable citations.

## Do's and Don'ts

- Do use compact document typography with modest blue accents.
- Do lead with current research, verified references, and useful academic contact.
- Do preserve original public paths and archival source bodies.
- Do follow final publisher citation metadata rather than assuming the DOI year.
- Do update runtime tokens and this document together and verify desktop and phone rendering.
- Don't restore oversized slogans, research cards, pill labels, or large buttons.
- Don't infer authorship roles, fabricate achievements, or add empty sections.
- Don't hide scrollbars or use color as the only interaction cue.
- Don't require a client framework or server for static academic content.
