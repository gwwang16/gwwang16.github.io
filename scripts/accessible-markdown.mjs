import { readFileSync, existsSync } from 'node:fs';
import { resolve, basename } from 'node:path';
import { imageSize } from 'image-size';

export const animationPosters = {
  '/images/portfolio/lane.gif': '/images/portfolio/lane.png',
  '/images/portfolio/lane-detection/output.gif':
    '/images/portfolio/lane-detection/result.png',
  '/images/portfolio/deep-rl-arm.gif': '/images/portfolio/deep-rl-arm.png',
  '/images/portfolio/pick-place/pick-place.gif':
    '/images/portfolio/pick-place/pick-place.png',
  '/images/portfolio/behavior-clone/clone.gif':
    '/images/portfolio/behavior-clone/clone.png',
  '/images/portfolio/semantic-segmentation/semantic-segmentation.gif':
    '/images/portfolio/semantic-segmentation/semantic-segmentation.png',
};

export function accessibleMarkdown() {
  return (tree) => {
    function walk(parent) {
      if (!parent.children) return;
      parent.children = parent.children.map((node) => {
        walk(node);
        if (node.type !== 'element') return node;
        if (node.tagName === 'h1') node.tagName = 'h2';
        if (node.tagName === 'table') {
          return {
            type: 'element',
            tagName: 'div',
            properties: {
              className: ['table-scroll'],
              tabIndex: 0,
              role: 'region',
              ariaLabel: 'Technical data table',
            },
            children: [node],
          };
        }
        if (node.tagName !== 'img') return node;
        const original = node.properties.src;
        const poster = animationPosters[original];
        const source = poster ?? original;
        node.properties.src = source;
        node.properties.loading = 'lazy';
        node.properties.decoding = 'async';
        if (!node.properties.alt || node.properties.alt === 'alt text') {
          node.properties.alt = basename(original)
            .replace(/\.[^.]+$/, '')
            .replace(/[-_]/g, ' ');
        }
        if (source?.startsWith('/images/')) {
          const file = resolve('public', `.${source}`);
          if (!existsSync(file))
            throw new Error(`Archived image is missing: ${source}`);
          const size = imageSize(readFileSync(file));
          node.properties.width = size.width;
          node.properties.height = size.height;
          if (poster)
            node.properties.style = `aspect-ratio: ${size.width} / ${size.height}`;
        }
        if (!poster) return node;
        node.properties.dataAnimation = original;
        node.properties.dataPoster = poster;
        return {
          type: 'element',
          tagName: 'figure',
          properties: { className: ['animation-figure'] },
          children: [
            node,
            {
              type: 'element',
              tagName: 'button',
              properties: {
                type: 'button',
                className: ['animation-toggle'],
                ariaPressed: 'false',
                hidden: true,
              },
              children: [{ type: 'text', value: 'Play animation' }],
            },
          ],
        };
      });
    }
    walk(tree);
  };
}
