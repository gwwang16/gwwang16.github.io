import type { APIRoute } from 'astro';
import { selectedPapers } from '../lib/content';
import { bibtex } from '../lib/bibliography';
import { profile } from '../data/profile';

export const GET: APIRoute = async () => {
  const book = profile.book;
  const bookCitation = `@book{${book.id},
  title = {{${book.title}}},
  author = {${book.authors.map((author) => `{${author}}`).join(' and ')}},
  publisher = {${book.publisher}},
  year = {${book.year}},
  language = {chinese}
}`;
  return new Response(`${bookCitation}\n\n${bibtex(await selectedPapers())}`, {
    headers: { 'Content-Type': 'application/x-bibtex; charset=utf-8' },
  });
};
