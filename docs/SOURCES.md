# Academic content sources

Checked on **5 October 2026**. This is the source record for the refactor, not a claim that every publication or achievement is listed.

## Profile and research

- [Guizhou University faculty profile](https://mech.gzu.edu.cn/2026/0416/c23422a272715/page.htm), published 27 April 2026: name, position, institute leadership, public email and office, research directions, recruitment / Tsinghua co-supervision, funded projects, courses, book, and student competition outcomes.
- [ResearchGate: Guangwei Wang at Guizhou University](https://www.researchgate.net/profile/Guangwei-Wang-3): publication discovery and identity cross-check. Other same-name profiles were not used.
- [Google Scholar](https://scholar.google.com/citations?user=2y82dCoAAAAJ&hl=en): original author-profile link preserved. Direct retrieval returned HTTP 429, so no citation counts or author-role claims were extracted from it.
- [ORCID](https://orcid.org/0000-0002-1794-0619): original identifier preserved; also matches the publisher metadata for the verified papers.
- [Faculty portrait](https://mech.gzu.edu.cn/_upload/tpl/0d/dd/3549/template3549/assets/teacher/WangGuangWei.jpg): current image from the faculty profile. The source photo is preserved and Astro generates compressed responsive images.

English descriptions paraphrase the published research scope. English institution/project/course/book titles are descriptive translations where the faculty page supplies only Chinese wording. Original Chinese funded-project titles are shown alongside those translations.

## Selected publications

The site includes **14** verified major papers, with **5** confirmed first-author entries and **11** confirmed corresponding-author entries (two appear in both groups). Each JSON record contains source URLs and a verification note. Correspondence is never inferred from last authorship.

| Record                                                               | Author-role evidence                                                                                                                                     | Citation metadata                                                                                                                                      |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| High-altitude adaptive net power optimization, 2026                  | Faculty profile explicitly marks Wang as corresponding                                                                                                   | [Energy](https://doi.org/10.1016/j.energy.2026.141442), Crossref: 359, 141442                                                                          |
| Prescribed-time sliding mode control, 2026                           | Faculty profile explicitly marks Wang as corresponding                                                                                                   | [Sage](https://doi.org/10.1177/10775463261450890), Crossref: advance online publication                                                                |
| Multi-terrain RRT*, 2026                                             | [Springer corresponding-author section](https://link.springer.com/article/10.1007/s42154-025-00431-2) and faculty profile                                | Same publisher / Crossref: advance online publication                                                                                                  |
| Dynamic obstacle avoidance parking trajectory planning, 2026         | [Journal author notes](https://www.journalase.com/CN/10.3969/j.issn.1674-8484.2026.04.008) explicitly identify 王广玮                                    | 17(4): 494–502                                                                                                                                         |
| Autonomous driving on mountain roads, 2025                           | Faculty profile explicitly marks Wang as corresponding                                                                                                   | [SciOpen](https://www.sciopen.com/article/10.26599/JICV.2026.9210069), Crossref: 8(4), 9210069                                                         |
| Energy management of fuel-cell hybrid bus, 2024                      | Faculty profile explicitly marks Wang as corresponding                                                                                                   | [Energy](https://doi.org/10.1016/j.energy.2024.133313), Crossref: 311, 133313; complete seven-author list replaces the faculty page’s abbreviated list |
| Adjustable compliant constant-force microgripper, 2024               | [MDPI author mark](https://www.mdpi.com/2072-666X/15/1/52) / [PMC author note](https://pmc.ncbi.nlm.nih.gov/articles/PMC10818475/)                       | Publisher citation: 15(1), 52                                                                                                                          |
| Real-time LiDAR point-cloud semantic segmentation, 2024              | [Journal author notes](https://www.journalase.com/CN/10.3969/j.issn.1674-8484.2024.04.016) explicitly identify 王广玮 (fourth author)                    | 15(4): 591–601                                                                                                                                         |
| Automatic optimization for compliant constant-force mechanisms, 2023 | [MDPI author mark and correspondence](https://www.mdpi.com/2076-0825/12/2/61)                                                                            | Publisher / Crossref: 12(2), 61                                                                                                                        |
| Robust nanopositioning tracking, 2023                                | [Published paper, first page](https://bwang-ccny.github.io/files/Papers/JVC-2023.pdf) confirms Wang first and explicitly identifies him as corresponding | Final issue: 29(15–16): 3809–3822                                                                                                                      |
| Fixed-time third-order sliding-mode control, 2021                    | [MDPI author list](https://www.mdpi.com/2227-7390/9/15/1770) confirms Wang first and corresponding                                                       | Publisher / Crossref: 9(15), 1770                                                                                                                      |
| Adaptive terminal sliding-mode control, 2018                         | [Publisher author list](https://doi.org/10.1002/asjc.1614) and original archive confirm Wang first                                                       | Crossref / final citation: 20(3): 1241–1252                                                                                                            |
| Precision position/force microinjection control, 2017                | [IEEE author list](https://ieeexplore.ieee.org/document/7912296/) confirms Wang first                                                                    | 22(4): 1744–1754                                                                                                                                       |
| Force-feedback microinjection system, 2017                           | [Taylor & Francis author list](https://www.tandfonline.com/doi/full/10.1080/01691864.2017.1362996) confirms Wang first; Xu is corresponding              | 31(23–24): 1349–1359                                                                                                                                   |

Crossref metadata can be checked at `https://api.crossref.org/works/<DOI>`. The maintained JSON uses complete author lists; abbreviated names are a rendering choice only.

## Citation-year corrections

- Micromachines microgripper: appeared online 26 December 2023; the publisher’s citation is **2024**, volume 15, issue 1.
- Robust tracking: appeared online 16 June 2022; the final Journal of Vibration and Control issue is **2023**, volume 29, issues 15–16.
- Mountain-road driving: the publisher’s issue is **2025**, volume 8, issue 4, despite “2026” in the DOI string.
- Adaptive terminal control: appeared online in 2017; the final Asian Journal of Control citation is **2018**.
- Advanced Robotics microinjection: final pages **1349–1359**, replacing the archive’s preliminary “1–11” in the current bibliography.
- Papers without assigned volume/pages are explicitly marked as advance online publications; missing metadata is not invented.

## Archive and exclusions

Eight early-work summaries are grounded in the original seven portfolios and six dated tutorials; the Home Service Robot tutorial supplies the eighth project. Original Markdown bodies, code links, images, historical CV, and public paths are retained. Long tutorials are condensed in the public interface, and their URLs redirect to matching summaries.

The original site supplies appointment dates. The current faculty page confirms the positions and institutions without specifying those dates.

Other ResearchGate papers and preprints are not automatically imported. Coauthorship or last-author position alone does not establish correspondence. The two historical unpublished/draft records remain private. No publication counts, impact factors, citation metrics, patent claims, or future achievements are invented or automatically scraped into the UI.
