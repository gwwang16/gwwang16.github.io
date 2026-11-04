export const profile = {
  name: 'Guangwei Wang',
  chineseName: '王广玮',
  role: 'Associate Professor',
  department: 'School of Mechanical Engineering',
  university: 'Guizhou University',
  location: 'Guiyang, China',
  site: 'https://www.guangwei.wang',
  emailDisplay: 'gwwang [at] gzu.edu.cn',
  office: 'Room 632, School of Mechanical Engineering, West Campus',
  description:
    'Guangwei Wang, Associate Professor at Guizhou University. Research in intelligent vehicles, precision motion control, and compliant mechanisms.',
  recruitment:
    'Master’s students and postdoctoral researchers are welcome to contact me by email.',
  analyticsId: 'G-141RDTBDLJ',
  links: {
    github: 'https://github.com/gwwang16',
    scholar: 'https://scholar.google.com/citations?user=2y82dCoAAAAJ&hl=en',
    researchgate: 'https://www.researchgate.net/profile/Guangwei-Wang-3',
    university: 'https://mech.gzu.edu.cn/2026/0416/c23422a272715/page.htm',
    orcid: 'https://orcid.org/0000-0002-1794-0619',
  },
  grants: [
    {
      period: '2026–2028',
      funder: 'National Natural Science Foundation of China',
      title:
        'Minimally invasive safety arbitration control for intelligent vehicles under multiple time-varying constraints',
      originalTitle: '面向时变多约束的智能车辆最小侵入式安全仲裁控制方法',
    },
    {
      period: '2023–2026',
      funder: 'National Natural Science Foundation of China',
      title:
        'Structural optimization and control of compliant constant-force microgrippers for highly dynamic micro/nano manipulation',
      originalTitle: '面向大动态微纳操作的柔性恒力微夹钳结构优化与控制方法研究',
    },
    {
      period: '2026–2029',
      funder: 'Guizhou Provincial Science and Technology Support Program',
      title:
        'Integrated active collision avoidance for autonomous buses under multiple physical constraints',
      originalTitle: '基于多重物理约束的无人驾驶巴士全向主动避险系统研发',
    },
    {
      period: '2025–2026',
      funder: 'Industry collaboration',
      title: 'Safety arbitration control system for road vehicles',
      originalTitle: '车辆行驶安全仲裁控制系统',
    },
  ],
  courses: [
    {
      level: 'Undergraduate',
      title: 'Automobile Theory',
      chineseTitle: '汽车理论',
    },
    {
      level: 'Undergraduate',
      title: 'Automobile Construction',
      chineseTitle: '汽车构造',
    },
    {
      level: 'Graduate',
      title: 'Intelligent and Connected Vehicles',
      chineseTitle: '智能网联汽车',
    },
  ],
  mentoring: [
    {
      year: '2025',
      result: 'National first prize',
      event: 'RAICOM Robotics Developer Competition · ROS virtual simulation',
    },
    {
      year: '2024',
      result: 'Second place nationally',
      event: 'Onsite Autonomous Driving Algorithm Challenge · Parking track',
    },
  ],
  book: {
    id: 'vehicle-platoon-control-2023',
    title: '智能车辆队列纵向与横向控制',
    authors: ['赵津', '王广玮', '石晴'],
    publisher: '重庆大学出版社',
    year: '2023',
  },
  experience: [
    {
      period: 'Dec 2022 — Present',
      role: 'Associate Professor',
      institution: 'Guizhou University',
      department: 'School of Mechanical Engineering',
    },
    {
      period: 'Mar 2023 — Apr 2025',
      role: 'Postdoctoral Researcher',
      institution: 'Tsinghua University',
      department: 'School of Vehicle and Mobility',
    },
    {
      period: 'Apr 2019 — Dec 2022',
      role: 'Lecturer',
      institution: 'Guizhou University',
      department: 'School of Mechanical Engineering',
    },
    {
      period: 'Aug 2015 — Jul 2018',
      role: 'Ph.D. in Electromechanical Engineering',
      institution: 'University of Macau',
      department: '',
    },
  ],
} as const;
