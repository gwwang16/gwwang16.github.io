import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse } from 'yaml';

test('documented design tokens match the canonical runtime stylesheet', () => {
  const doc = readFileSync('DESIGN.md', 'utf8');
  const frontmatter = parse(doc.match(/^---\n([\s\S]*?)\n---/)[1]);
  const stylesheet = readFileSync('src/styles/global.css', 'utf8');
  const root = stylesheet.match(/:root\s*\{([\s\S]*?)\}/)[1];
  const tokens = Object.fromEntries(
    [...root.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map(([, key, value]) => [
      key,
      value.trim(),
    ]),
  );
  for (const [key, value] of Object.entries(frontmatter.colors))
    assert.equal(tokens[`--color-${key}`], value, key);
  for (const [key, value] of Object.entries(frontmatter.rounded))
    assert.equal(tokens[`--radius-${key}`], value, key);
  for (const [key, value] of Object.entries(frontmatter.spacing))
    assert.equal(tokens[`--${key}`], value, key);
  for (const [key, value] of Object.entries(frontmatter.typography))
    assert.equal(tokens[`--font-${key}`], value.fontFamily, key);
});
