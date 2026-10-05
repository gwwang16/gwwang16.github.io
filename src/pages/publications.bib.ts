import type { APIRoute } from 'astro';
import { selectedPapers } from '../lib/content';
import { bibtex } from '../lib/bibliography';

export const GET: APIRoute = async () =>
  new Response(bibtex(await selectedPapers()), {
    headers: { 'Content-Type': 'application/x-bibtex; charset=utf-8' },
  });
