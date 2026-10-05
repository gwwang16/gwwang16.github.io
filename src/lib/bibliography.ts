import type { CollectionEntry } from 'astro:content';

const escape = (value: string) => value.replaceAll('&', '\\&');

export function bibtex(papers: CollectionEntry<'papers'>[]) {
  return (
    papers
      .map(({ id, data }) => {
        const fields: Record<string, string> = {
          title: `{${data.title}}`,
          author: data.authors
            .map((author) => {
              const parts = author.split(' ');
              return parts.length > 1
                ? `${parts.at(-1)}, ${parts.slice(0, -1).join(' ')}`
                : author;
            })
            .join(' and '),
          journal: data.journal,
          year: String(data.year),
          doi: data.doi,
          url: `https://doi.org/${data.doi}`,
        };
        if (data.volume) fields.volume = data.volume;
        if (data.issue) fields.number = data.issue;
        if (data.pages)
          fields.pages = data.pages
            .replaceAll('–', '--')
            .replaceAll(/(?<!-)-(?!-)/g, '--');
        return `@article{${id},\n${Object.entries(fields)
          .map(([key, value]) => `  ${key} = {${escape(value)}}`)
          .join(',\n')}\n}`;
      })
      .join('\n\n') + '\n'
  );
}
