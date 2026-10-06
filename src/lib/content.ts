import { getCollection } from 'astro:content';

export async function selectedPapers() {
  return (await getCollection('papers')).sort(
    (a, b) =>
      b.data.year - a.data.year ||
      (a.data.featuredOrder ?? 99) - (b.data.featuredOrder ?? 99) ||
      a.id.localeCompare(b.id),
  );
}

export async function researchAreas() {
  return (await getCollection('research')).sort(
    (a, b) => a.data.order - b.data.order,
  );
}

export async function publishedPapers() {
  return (await getCollection('publications', ({ data }) => !data.draft)).sort(
    (a, b) =>
      (b.data.year ?? b.data.date.getFullYear()) -
        (a.data.year ?? a.data.date.getFullYear()) ||
      b.data.date.getTime() - a.data.date.getTime(),
  );
}

export async function allProjects() {
  return (await getCollection('projects')).sort((a, b) =>
    a.data.title.localeCompare(b.data.title, 'en'),
  );
}

export async function earlyWork() {
  const projects = (await allProjects()).map(({ data }) => ({
    id: data.slug,
    title: data.title,
    description: data.description,
    image: data.image,
    imageAlt: data.imageAlt,
    repository: data.repository,
    demo:
      data.slug === 'search-and-sample-return'
        ? 'https://youtu.be/SUSzm_10sPE'
        : undefined,
  }));
  projects.push({
    id: 'home-service-robot',
    title: 'ROS service robot navigation',
    description:
      'SLAM, localization, and waypoint navigation in a simulated indoor environment.',
    image: '/images/portfolio/home-service-robot/results.jpg',
    imageAlt:
      'ROS mapping and navigation display beside the simulated indoor environment.',
    repository: 'https://github.com/gwwang16/Home-Service-Robot',
    demo: undefined,
  });
  return projects.sort((a, b) => a.title.localeCompare(b.title, 'en'));
}

export function earlyWorkForArticle(permalink: string) {
  const targets: Record<string, string> = {
    '/posts/pr2-3d-perception': '3d-perception',
    '/posts/2017/behavioral-cloning': 'behavioral-cloning',
    '/posts/2018/lane-detection': 'lane-detection',
    '/posts/2018/iiwa-kinematics': 'iiwa-kinematics',
    '/posts/2018/home-service-robot': 'home-service-robot',
    '/posts/2018/lyft-challenge': 'lyft-perception',
  };
  const id = targets[permalink.replace(/\/$/, '')];
  if (!id) throw new Error(`Unmapped early article: ${permalink}`);
  return `/projects/#${id}`;
}
