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
  section-space: '32px'
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

Public copy states academic facts directly. Keep appointments in the timeline and funding in its own section. Research overviews describe problems and methods supported by their citations. Omit editing/migration narratives, instructions explaining ordinary links, repeated profile facts, and redundant technology tags. Authorship and citation-year verification notes belong in source records. The privacy page describes the services used by the current site.

Token ownership uses Model B: `src/styles/global.css` is the canonical runtime source. This document mirrors accepted values; update both when changing a token. Shared layouts and citation components keep every public page consistent.

## Colors

White carries all main content. The name and section headings use the dark blue accent; links use a slightly brighter blue with visible underlines. Hover and keyboard focus deepen link color and increase underline thickness from 1px to 2px. Grey-blue rules separate sections, navigation, and the footer. Charcoal body text and muted secondary metadata support long-form reading. Pale surfaces are reserved for technical tables, code, and animation controls, rather than alternating homepage bands.

`colors.<name>` maps to `--color-<name>`. Scrollbar tokens define thumb, track, hover, and active colors with standards-based properties and WebKit fallbacks. Forced colors use system colors, and focus has a distinct blue outline.

## Typography

Use the native Arial / Helvetica stack with script-capable fallbacks. No web fonts are required. Body text and references share 16px / 1.6. Navigation uses 16px text at weight 500, with Home at weight 600. The homepage name is 32px, other page titles are 28px, section headings are 20px, and subsections are 16px. The name reduces to 26px on phones; bibliography text remains 16px.

`typography.display`, `.body`, and `.mono` map to their font custom properties. Avoid exaggerated title sizes, decorative punctuation, uppercase section labels, and widely spaced lettering. Metadata uses 13–14px. References wrap naturally, abbreviate author names only when rendered, bold the researcher’s name, and mark verified correspondence with an asterisk. Full authors remain in data and BibTeX export.

## Layout

Use a centered 960px document with a 24px minimum desktop margin and 16px phone margin. Section spacing is 32px. The body is a single column; small date columns organize funding and appointments on desktop and stack on phones. Heading rules create structure without cards or colored panels.

The profile combines name, position, institution, readable obfuscated email, a short office location, and academic links. Keep contact details together near the top; omit a separate statement of supervision or administrative responsibilities. A 120px portrait sits beside the identity block, reducing to 100px or 80px on narrow screens. Preserve its aspect ratio and use optimized responsive assets. Navigation is a 64px text row with a thin bottom rule and restrained blue link indicators. At 800px it becomes a 52px native disclosure that expands in document flow.

The homepage sequence is identity and contact → research interests → representative references → funded projects → education and experience → teaching → Join us. Use Join us consistently for the recruitment heading and desktop/mobile navigation; the invitation names master’s students and postdoctoral researchers. Preserve the existing `#prospective-students` anchor. The separate publications page shows selected papers first, followed by books, under distinct headings. Earlier projects have a footer entry; their rows use small static thumbnails, short descriptions, and source/demo links.

`spacing.page-width` and `.section-space` map to their runtime custom properties. Document scrolling owns the page; code, tables, and long formulas may scroll locally. Hash-linked references and summaries remain visible.

## Elevation & Depth

Use thin rules and spacing. No shadows, large colored bands, floating navigation, pill tags, or promotional action panels. A compact text footer closes the document. Content remains visible without client requests, animation, or loading placeholders.

## Shapes

The portrait is square-cornered. The 2px control radius applies only to technical animation controls and scrollbar styling. These map from `rounded.panel` / `.control` to their runtime tokens. Academic navigation and profile links are ordinary text links.

## Components

Research interests are plain list entries with a linked title and short summary. Publication rows are shared across the homepage, bibliography, and research pages. Keep verified correspondence asterisks without a separate explanatory legend. The same records generate BibTeX. Author roles require evidence; last authorship alone is insufficient. Funded projects show dates, titles, funders, and original Chinese names without unsupported role claims.

The academic appointment list preserves the original dates and overlapping positions. Teaching is a short course list; student competition records remain available through the university profile. Books use a full reference with the original Chinese title, author list, Chinese publisher name, and year. Highlight the researcher's name consistently with paper citations. BibTeX includes both the book and papers.

Email replaces `@` with readable `[at]` text and keeps the domain's dots intact, without JavaScript, a plain `mailto` link, or a full address in person metadata. The privacy page links to this contact block. This is lightweight protection against simple harvesting, not an assertion that automated extraction is impossible. The short office location appears beside email. Prospective students receive a concise invitation to contact by email; the university profile link stays in the top profile block.

Earlier-project summaries are readable without JavaScript and use small responsive WebP thumbnails generated from the preserved original images. Old tutorials and portfolio URLs redirect to matching summary anchors; original bodies remain in source. Historical paper URLs redirect to updated references. Draft records do not generate public routes. The historical CV remains an asset; the current university page is the visible institutional reference.

The Markdown renderer supports highlighted code, math/MathML, accessible tables, static posters, and opt-in animation controls for future research content. Formula styles and animation scripts load only when the rendered content needs them. Animation failure returns to a poster, and reduced-motion changes pause playback.

Navigation links use muted text at rest and the deeper primary-hover blue on hover, keyboard focus, and the current page. A 2px underline expands from the left on interaction and remains visible for the current page; font weights stay stable between these states. Desktop links have a 24px gap. Home is current on the homepage, and Research is current on research detail pages. The local `--navigation-duration` in `.header-inner` owns the 180ms color, underline, and menu-indicator transitions; the global reduced-motion rule disables their duration. The header stays in document flow.

The native mobile menu uses a decorative two-line indicator that becomes a cross when expanded. It supports Escape, outside-click closing, and navigation closing without claiming modal behavior. Menu links keep a minimum 44px target height. Supplementary source/demo icons are hidden from assistive technology. The 404 page offers plain home and earlier-project recovery links. Print styles remove navigation and preserve readable citations.

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
