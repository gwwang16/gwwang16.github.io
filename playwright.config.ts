import { defineConfig } from 'playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  workers: process.env.CI ? 2 : 3,
  retries: process.env.CI ? 1 : 0,
  timeout: 45_000,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:4321',
    headless: true,
    viewport: { width: 1440, height: 1000 },
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run preview -- --port 4321 --ignore-lock',
    url: 'http://127.0.0.1:4321',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
    env: { ASTRO_TELEMETRY_DISABLED: '1' },
  },
});
