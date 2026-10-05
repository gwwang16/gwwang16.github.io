---
version: alpha
name: Guangwei Wang
description: A restrained academic profile centered on verified research, reference-style citations, and concise early work.
colors:
  primary: '#176b80'
  primary-hover: '#115466'
  ink: '#193247'
  muted: '#516577'
  background: '#ffffff'
  paper: '#eef3f7'
  soft: '#f7f9fb'
  border: '#d7e1e9'
  accent: '#32796b'
  focus: '#0964cf'
typography:
  display:
    fontFamily: "'Manrope Variable', system-ui, sans-serif"
  body:
    fontFamily: "'Source Sans 3', 'Noto Sans SC', system-ui, sans-serif"
    fontSize: '18px'
    lineHeight: '1.65'
  mono:
    fontFamily: "'SFMono-Regular', Consolas, monospace"
rounded:
  control: '6px'
  panel: '14px'
spacing:
  page-width: '1120px'
  section-space: '96px'
components:
  button:
    height: '48px'
    rounded: '{rounded.control}'
    backgroundColor: '{colors.primary}'
    textColor: '{colors.background}'
  button-hover:
    backgroundColor: '{colors.primary-hover}'
    textColor: '{colors.background}'
  research-panel:
    rounded: '{rounded.panel}'
    backgroundColor: '{colors.background}'
    textColor: '{colors.ink}'
  secondary-text:
    textColor: '{colors.muted}'
  portrait-panel:
    backgroundColor: '{colors.background}'
    textColor: '{colors.muted}'
  publication-section:
    backgroundColor: '{colors.soft}'
    textColor: '{colors.ink}'
  background-section:
    backgroundColor: '{colors.paper}'
    textColor: '{colors.ink}'
  tag:
    backgroundColor: '{colors.background}'
    textColor: '{colors.muted}'
  divider:
    backgroundColor: '{colors.border}'
  identity-accent:
    textColor: '{colors.accent}'
  focus-ring:
    backgroundColor: '{colors.focus}'
---

# Guangwei Wang Design System

## Overview

This is a public academic profile for research collaborators and prospective postgraduate students. Its priority is current research and verified first-author / corresponding-author publications. Student-era ROS and autonomous-driving work is a compact supporting collection. The site follows a quiet academic document style: a current faculty portrait, clear typography, pale blue surfaces, reference-style citations, and direct links to evidence.

English is the existing site language. The Chinese name, original funded-project titles, course names, and Chinese-journal citations retain their native wording and appropriate language attributes. Current facts come from the university profile and verified publisher records, documented in `docs/SOURCES.md`. Do not add invented metrics, vague promotional slogans, empty future-results sections, or dashboard conventions.

Token ownership uses Model B: `src/styles/global.css` is the canonical runtime token source. This document mirrors accepted values and explains their use. `tests/design.test.mjs` checks the mapping. All content pages share `BaseLayout`, `Header`, `Footer`, and citation components.

## Colors

White is the document surface. Paper blue groups academic background and the footer; the softer surface groups references and contact. Navy carries headings, muted slate carries secondary text, and teal identifies links and primary actions. Green is a restrained identity accent. Blue focus rings are distinct from brand color. Borders separate content and never carry meaning alone.

`colors.<name>` maps to `--color-<name>`. Global scrollbar tokens define thumb, track, hover, and active colors with standards-based properties and WebKit fallbacks. Forced colors use system colors.

## Typography

Manrope is used for the name, headings, and compact research labels. Source Sans 3 provides the body and bibliography reading face. Both are self-hosted; native script-capable fallbacks cover Chinese text. The body uses 18px / 1.65, references use 17px / 1.75, and references reduce to 16px on phones.

`typography.display`, `.body`, and `.mono` map to the corresponding font custom properties. Paper titles wrap naturally. Author names are abbreviated only when rendered; complete author lists remain in data and export. The researcher’s name is bold and verified correspondence uses an asterisk with a visible legend. Short uppercase labels organize sections without dominating content.

## Layout

The 1120px document grid has generous desktop margins and a 20px phone margin. Section spacing is 96px on desktop, 66px on tablets, and 56px on phones. At 800px the header becomes a native, nonmodal disclosure menu; at 600px the hero, research, teaching, and background columns stack.

The homepage sequence is identity → research → representative references → funded projects → background → teaching → concise earlier work → contact. The portrait reserves a fixed aspect ratio and uses responsive optimized assets. The academic timeline preserves the original dates and overlapping appointments. The bibliography is a numbered reference list rather than large cards. Earlier-work rows combine a small static preview, a short description, and source/demo links.

`spacing.page-width` and `.section-space` map to their CSS custom properties. Document scrolling owns the page; code, tables, and long formulas may have local overflow. Sticky navigation has corresponding scroll padding. Hash-linked citations and work summaries remain visible below the header.

## Elevation & Depth

Static content uses borders and tonal surfaces without shadows. Only the mobile navigation popover has a restrained shadow. The header is opaque and sticky. No content depends on scroll-triggered visibility, loading placeholders, or client requests.

## Shapes

Controls use the 6px control radius and research/media panels use the 14px panel radius. These map from `rounded.control` / `.panel` to their runtime tokens. Circular details are limited to the timeline and identity accents.

## Components

Research cards link to Markdown overviews and verified related references. Publication rows share one component across the homepage, publication list, and research pages. The same records generate BibTeX. Author roles must have source evidence; last authorship alone is insufficient.

Earlier-work summaries are fully readable without JavaScript. Old tutorials and portfolio URLs redirect to matching summary anchors; original text stays in source rather than occupying long public pages. Historical paper URLs redirect to updated citations. Draft records do not generate public routes. The historical CV is retained as an asset, while the current faculty profile is the visible institutional reference.

The Markdown renderer supports build-time highlighted code, math/MathML, accessible tables, and static posters with opt-in animation controls for future technical research content. Media failure returns to a poster and reduced-motion changes pause animations.

SVG icons supplement visible labels and decorative icons are hidden from assistive technology. The native menu supports Escape, outside-click closing, and navigation closing without claiming modal behavior. Hover feedback respects reduced motion. The 404 page offers home and earlier-work recovery links.

## Do's and Don'ts

- Do lead with current research, verified publications, and useful academic contact.
- Do preserve original public paths and archival source bodies.
- Do keep references consistent with their final publisher citation, not an assumed filename or DOI year.
- Do update runtime tokens and this document together, then run drift checks.
- Don't infer correspondence, fabricate achievements, or introduce unfinished-result cards.
- Don't hide scrollbars or use animation/color as the only state cue.
- Don't require a client framework or server for static academic content.
