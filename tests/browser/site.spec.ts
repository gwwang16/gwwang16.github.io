import { test, expect } from 'playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';
const migration = JSON.parse(
  readFileSync('docs/content-migration.json', 'utf8'),
) as { entries: { route: string; draft: boolean }[] };
const papers = JSON.parse(
  readFileSync('src/content/selected-publications.json', 'utf8'),
) as { id: string; title: string; doi: string; roles: string[] }[];
const routes = [
  '/',
  '/projects/',
  '/publications/',
  '/research/intelligent-vehicles/',
  '/research/precision-manipulation/',
  '/sitemap/',
  '/terms/',
  '/404.html',
  ...migration.entries
    .filter((entry) => !entry.draft)
    .map((entry) => entry.route),
];

test.beforeEach(async ({ context }) => {
  await context.route('**/*googletagmanager.com/**', (route) =>
    route.fulfill({ status: 200, body: '' }),
  );
  await context.route('**/*google-analytics.com/**', (route) =>
    route.fulfill({ status: 204, body: '' }),
  );
});

test('home renders current academic content and supports keyboard skip navigation', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(
    page.getByRole('heading', { name: 'Guangwei Wang', exact: true }),
  ).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  await page.keyboard.press('Tab');
  await expect(
    page.getByRole('link', { name: 'Skip to content' }),
  ).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('#main-content')).toBeFocused();
  await expect(page.locator('.publication-item')).toHaveCount(6);
  await expect(page.locator('.grant-item')).toHaveCount(4);
  await expect(page.locator('.experience-list h3')).toHaveCount(4);
  await expect(
    page.getByRole('link', { name: 'gwwang@gzu.edu.cn' }),
  ).toHaveAttribute('href', 'mailto:gwwang@gzu.edu.cn');
  expect(
    await page
      .locator('.profile-visual img')
      .evaluate(
        (image: HTMLImageElement) => image.complete && image.naturalWidth > 0,
      ),
  ).toBe(true);
  expect(errors).toEqual([]);
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.screenshot({
    path: 'test-results/home-desktop.png',
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: 'test-results/home-mobile.png',
    fullPage: true,
  });
});

test('the bibliography includes only selected first or corresponding author papers with DOI links', async ({
  page,
  request,
}) => {
  await page.goto('/publications/');
  await expect(page.locator('.publication-item')).toHaveCount(papers.length);
  for (const paper of papers) {
    const row = page.locator(`#${paper.id}`);
    await expect(
      row.getByRole('link', { name: paper.title, exact: true }),
    ).toHaveAttribute('href', `https://doi.org/${paper.doi}`);
    await expect(row.locator('.self-author')).toHaveCount(1);
    await expect(row.locator('sup')).toHaveCount(
      paper.roles.includes('corresponding') ? 1 : 0,
    );
  }
  const response = await request.get('/publications.bib');
  expect(response.status()).toBe(200);
  const citations = await response.text();
  for (const paper of papers) expect(citations).toContain(paper.doi);
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download BibTeX' }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('publications.bib');
  expect(await download.failure()).toBeNull();
  await page.screenshot({
    path: 'test-results/publications-desktop.png',
    fullPage: true,
  });
});

test('research overviews connect current work to the publication record', async ({
  page,
}) => {
  await page.goto('/');
  const overview = page
    .getByRole('link', { name: 'Research overview', exact: true })
    .first();
  await overview.focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/research\/intelligent-vehicles\/$/);
  await expect(
    page.getByRole('heading', { name: 'Intelligent vehicles', exact: true }),
  ).toBeVisible();
  await expect(page.locator('.related-papers .publication-item')).toHaveCount(
    6,
  );
  await expect(
    page.getByRole('heading', { name: 'Current projects', exact: true }),
  ).toBeVisible();
});

test('early student work is condensed into one collection with source links', async ({
  page,
}) => {
  await page.goto('/projects/');
  await expect(page.locator('.early-work-item')).toHaveCount(8);
  for (const project of await page.locator('.early-work-item').all()) {
    expect((await project.innerText()).length).toBeLessThan(600);
  }
  await expect(
    page.getByRole('link', { name: 'Source code for Home Service Robot' }),
  ).toHaveAttribute('href', 'https://github.com/gwwang16/Home-Service-Robot');
  await page.screenshot({
    path: 'test-results/earlier-work-desktop.png',
    fullPage: true,
  });
});

test('mobile navigation opens, closes with Escape, and follows a link', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.locator('.mobile-navigation summary').click();
  await expect(
    page.getByRole('navigation', { name: 'Mobile navigation' }),
  ).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('.mobile-navigation')).not.toHaveAttribute('open');
  await expect(page.locator('.mobile-navigation summary')).toBeFocused();
  await page.locator('.mobile-navigation summary').click();
  await page
    .getByRole('navigation', { name: 'Mobile navigation' })
    .getByRole('link', { name: 'Publications', exact: true })
    .click();
  await expect(page).toHaveURL(/\/publications\/$/);
  await expect(page.locator('.mobile-navigation')).not.toHaveAttribute('open');
});

test('academic content, native menu, and archive redirects work without JavaScript', async ({
  browser,
}) => {
  const context = await browser.newContext({
    baseURL: 'http://127.0.0.1:4321',
    javaScriptEnabled: false,
    viewport: { width: 390, height: 844 },
  });
  await context.route('**/*googletagmanager.com/**', (route) =>
    route.fulfill({ status: 200, body: '' }),
  );
  const page = await context.newPage();
  await page.goto('/publications/');
  await expect(page.locator('.publication-item')).toHaveCount(14);
  await page.locator('.mobile-navigation summary').click();
  await expect(
    page.getByRole('navigation', { name: 'Mobile navigation' }),
  ).toBeVisible();
  await page.goto('/posts/2018/home-service-robot/');
  await page.waitForURL('http://127.0.0.1:4321/projects/#home-service-robot');
  await expect(page.locator('.early-work-item')).toHaveCount(8);
  await context.close();
});

test('legacy URLs resolve to the relevant summary and drafts stay private', async ({
  page,
  request,
}) => {
  for (const [alias, destination] of [
    ['/about.html', '/'],
    ['/about/', '/'],
    ['/blog/', '/'],
    ['/portfolio/5-deep-rl-arm/', '/projects/#deep-rl-robot-arm'],
    ['/projects/deep-rl-robot-arm/', '/projects/#deep-rl-robot-arm'],
    ['/posts/2018/lyft-challenge/', '/projects/#lyft-perception'],
    [
      '/publication/2017-08-14-advanced-robotics/',
      '/publications/#microinjection-feedback-2017',
    ],
    ['/notes/', '/projects/'],
  ]) {
    await page.goto(alias);
    await page.waitForURL(`http://127.0.0.1:4321${destination}`);
  }
  for (const entry of migration.entries.filter((entry) => entry.draft)) {
    expect((await request.get(entry.route)).status()).toBe(404);
  }
});

for (const width of [320, 768, 1440]) {
  test(`all public pages fit ${width}px without missing images or runtime errors`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.setViewportSize({ width, height: 900 });
    for (const route of [...new Set(routes)]) {
      const response = await page.goto(route);
      expect(response?.status(), route).toBeLessThan(400);
      await page.locator('footer').waitFor();
      await page.evaluate(() => {
        for (const image of document.images) image.loading = 'eager';
      });
      await page.locator('footer').scrollIntoViewIfNeeded();
      await page.waitForFunction(() =>
        [...document.images].every((image) => image.complete),
      );
      const dimensions = await page.evaluate(() => ({
        width: document.documentElement.clientWidth,
        content: document.documentElement.scrollWidth,
      }));
      expect(dimensions.content, route).toBeLessThanOrEqual(
        dimensions.width + 1,
      );
      const failedImages = await page.evaluate(() =>
        [...document.images]
          .filter((image) => image.naturalWidth === 0)
          .map((image) => image.src),
      );
      expect(failedImages, route).toEqual([]);
      await expect(page.locator('main')).toBeVisible();
      await expect(page.locator('footer')).toBeVisible();
    }
    expect(errors).toEqual([]);
  });
}

for (const route of [
  '/',
  '/projects/',
  '/publications/',
  '/research/intelligent-vehicles/',
  '/research/precision-manipulation/',
  '/terms/',
  '/404.html',
]) {
  test(`automated accessibility checks pass for ${route}`, async ({ page }) => {
    await page.goto(route);
    await page.evaluate(() => document.fonts.ready);
    const result = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze();
    expect(result.violations).toEqual([]);
  });
}
