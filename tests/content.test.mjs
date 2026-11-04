import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { parse } from 'yaml';

const papers = JSON.parse(
  readFileSync('src/content/selected-publications.json', 'utf8'),
);

test('paper IDs and DOIs are unique and research references exist', () => {
  const ids = new Set(papers.map(({ id }) => id));
  assert.equal(ids.size, papers.length, 'Duplicate paper ID');
  assert.equal(
    new Set(papers.map(({ doi }) => doi.toLowerCase())).size,
    papers.length,
    'Duplicate DOI',
  );
  for (const file of readdirSync('src/content/research').filter((name) =>
    name.endsWith('.md'),
  )) {
    const markdown = readFileSync(join('src/content/research', file), 'utf8');
    const data = parse(markdown.match(/^---\n([\s\S]*?)\n---/)[1]);
    for (const id of data.papers)
      assert.ok(ids.has(id), `${file}: unknown paper ${id}`);
  }
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

test('built pages have no broken internal links, assets, or fragment targets', () => {
  for (const file of htmlFiles('dist')) {
    const html = readFileSync(file, 'utf8');
    for (const [, url] of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
      if (url.startsWith('#')) {
        assert.ok(
          html.includes(`id="${decodeURIComponent(url.slice(1))}"`),
          `${file}: ${url}`,
        );
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
            `id="${decodeURIComponent(parsed.hash.slice(1))}"`,
          ),
          `${file}: ${url}`,
        );
    }
  }
});
