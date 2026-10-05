import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { parse } from 'yaml';
const manifest = JSON.parse(
  readFileSync('docs/content-migration.json', 'utf8'),
);

test('the migrated archive preserves all nineteen original document bodies', () => {
  assert.equal(manifest.entries.length, 19);
  for (const { destination, bodySha256 } of manifest.entries) {
    const body = readFileSync(destination, 'utf8')
      .replace(/^---\n[\s\S]*?\n---/, '')
      .trim();
    assert.equal(
      createHash('sha256').update(body).digest('hex'),
      bodySha256,
      destination,
    );
  }
});

test('every published migrated document has a generated route; drafts have none', () => {
  for (const { route, draft } of manifest.entries) {
    assert.equal(existsSync(join('dist', route, 'index.html')), !draft, route);
  }
  const sitemap = readFileSync('dist/sitemap-0.xml', 'utf8');
  for (const entry of manifest.entries.filter(({ draft }) => draft))
    assert.ok(!sitemap.includes(entry.route));
});

test('legacy files and the custom domain are preserved', () => {
  assert.equal(readFileSync('dist/CNAME', 'utf8').trim(), 'www.guangwei.wang');
  assert.ok(existsSync('dist/.nojekyll'));
  assert.ok(existsSync('dist/files/cv_gwwang_en.pdf'));
  assert.ok(existsSync('dist/images/profile.png'));
});

function htmlFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory()
      ? htmlFiles(join(dir, entry.name))
      : entry.name.endsWith('.html')
        ? [join(dir, entry.name)]
        : [],
  );
}

test('built pages have no missing internal links, assets, or fragment targets', () => {
  for (const file of htmlFiles('dist')) {
    const html = readFileSync(file, 'utf8');
    for (const [, url] of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
      if (url.startsWith('#')) {
        assert.ok(html.includes(`id="${url.slice(1)}"`), `${file}: ${url}`);
        continue;
      }
      if (!url.startsWith('/')) continue;
      const parsed = new URL(
        url.replaceAll('&amp;', '&'),
        'https://www.guangwei.wang',
      );
      const target = join('dist', decodeURIComponent(parsed.pathname));
      const resolved = existsSync(join(target, 'index.html'))
        ? join(target, 'index.html')
        : target;
      assert.ok(existsSync(resolved), `${file}: ${url}`);
      if (parsed.hash)
        assert.ok(
          readFileSync(resolved, 'utf8').includes(
            `id="${parsed.hash.slice(1)}"`,
          ),
          `${file}: ${url}`,
        );
    }
    if (!html.includes('http-equiv="refresh"')) {
      assert.match(html, /<html lang="en"/);
      assert.match(html, /<title>[^<]+<\/title>/);
      assert.match(html, /<meta name="description" content="[^\"]+"/);
      assert.match(html, /<link rel="canonical"/);
      assert.equal(
        (html.match(/<h1[\s>]/g) ?? []).length,
        1,
        `${file}: one primary heading`,
      );
    }
  }
});

test('old tutorials and paper pages redirect to condensed work and verified citations', () => {
  const sitemap = readFileSync('dist/sitemap-0.xml', 'utf8');
  for (const { route, draft } of manifest.entries) {
    if (draft || route === '/terms/') continue;
    const html = readFileSync(join('dist', route, 'index.html'), 'utf8');
    assert.match(html, /http-equiv="refresh"/, route);
    assert.match(html, /content="noindex, follow"/, route);
    assert.ok(!sitemap.includes(route.replace(/\/$/, '') + '/'), route);
    assert.ok(!html.includes('class="prose"'), route);
  }
});

const papers = JSON.parse(
  readFileSync('src/content/selected-publications.json', 'utf8'),
);

test('selected citations retain author-role evidence, unique DOI records, and final citation years', () => {
  assert.equal(new Set(papers.map((paper) => paper.id)).size, papers.length);
  assert.equal(
    new Set(papers.map((paper) => paper.doi.toLowerCase())).size,
    papers.length,
  );
  for (const paper of papers) {
    assert.ok(paper.roles.length > 0, paper.id);
    assert.ok(
      paper.roles.every((role) => ['first', 'corresponding'].includes(role)),
      paper.id,
    );
    assert.ok(
      paper.authors.some((name) => ['Guangwei Wang', '王广玮'].includes(name)),
      paper.id,
    );
    if (paper.roles.includes('first'))
      assert.equal(paper.authors[0], 'Guangwei Wang', paper.id);
    assert.equal(
      new URL(paper.sources.authorship).protocol,
      'https:',
      paper.id,
    );
    assert.equal(new URL(paper.sources.metadata).protocol, 'https:', paper.id);
    assert.ok(paper.sources.note.length > 20, paper.id);
    assert.match(paper.verifiedOn, /^\d{4}-\d{2}-\d{2}$/);
  }
  assert.equal(
    papers.find(({ id }) => id === 'adjustable-microgripper-2024').year,
    2024,
  );
  assert.equal(
    papers.find(({ id }) => id === 'robust-nanopositioning-2023').year,
    2023,
  );
  assert.equal(
    papers.find(({ id }) => id === 'mountain-driving-2025').year,
    2025,
  );
});

test('every selected paper appears in the citation page and BibTeX export', () => {
  const html = readFileSync('dist/publications/index.html', 'utf8');
  const bibliography = readFileSync('dist/publications.bib', 'utf8');
  assert.equal(
    (html.match(/class="publication-item"/g) ?? []).length,
    papers.length,
  );
  assert.equal((bibliography.match(/@article\{/g) ?? []).length, papers.length);
  for (const paper of papers) {
    assert.ok(html.includes(`id="${paper.id}"`), paper.id);
    assert.ok(html.includes(`https://doi.org/${paper.doi}`), paper.id);
    assert.ok(html.includes(paper.title), paper.id);
    assert.ok(bibliography.includes(`@article{${paper.id},`), paper.id);
    assert.ok(bibliography.includes(`doi = {${paper.doi}}`), paper.id);
  }
});

test('the homepage retains the complete academic timeline and keeps early work concise', () => {
  const home = readFileSync('dist/index.html', 'utf8');
  for (const fact of [
    'Dec 2022 — Present',
    'Mar 2023 — Apr 2025',
    'Apr 2019 — Dec 2022',
    'Aug 2015 — Jul 2018',
    'Associate Professor',
    'Postdoctoral Researcher',
    'Lecturer',
    'Ph.D. in Electromechanical Engineering',
    'gwwang@gzu.edu.cn',
  ])
    assert.ok(home.includes(fact), fact);
  const projects = readFileSync('dist/projects/index.html', 'utf8');
  assert.equal((projects.match(/class="early-work-item"/g) ?? []).length, 8);
  assert.ok(!projects.includes('class="prose"'));
});

test('research overviews reference existing verified papers and generate public routes', () => {
  const ids = new Set(papers.map(({ id }) => id));
  for (const file of readdirSync('src/content/research').filter((name) =>
    name.endsWith('.md'),
  )) {
    const markdown = readFileSync(join('src/content/research', file), 'utf8');
    const data = parse(markdown.match(/^---\n([\s\S]*?)\n---/)[1]);
    for (const id of data.papers)
      assert.ok(ids.has(id), `${file}: unknown paper ${id}`);
    assert.ok(
      existsSync(
        join('dist/research', file.replace(/\.md$/, ''), 'index.html'),
      ),
      file,
    );
  }
});
