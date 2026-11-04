---
title: Point-cloud object recognition
collection: portfolio
slug: 3d-perception
category: Robotics
image: /images/portfolio/pr2-perception/collision_map.png
imageAlt: PR2 simulation with segmented tabletop point clouds and a collision map.
description: Point-cloud segmentation and SVM-based object recognition for simulated
  PR2 pick-and-place tasks.
technologies:
- Point clouds
- PCL
- ROS
repository: https://github.com/gwwang16/RoboND-Perception-Project
article: /posts/pr2-3d-perception/
legacyPath: /portfolio/6-perception/
---

Given a cluttered tabletop scenario, perform object segmentation on 3D point cloud data using python-pcl to leverage the power of the Point Cloud Library, then identify target objects from a “Pick-List” in a particular order, pick up those objects and place them in corresponding drop boxes.