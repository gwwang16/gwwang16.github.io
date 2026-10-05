---
title: Robot arm reinforcement learning
collection: portfolio
slug: deep-rl-robot-arm
category: Robotics
image: /images/portfolio/deep-rl-arm.png
animation: /images/portfolio/deep-rl-arm.gif
description: Deep reinforcement learning for robotic manipulation on an NVIDIA Jetson
  TX2.
technologies:
- PyTorch
- OpenAI Gym
- Gazebo
repository: https://github.com/gwwang16/DeepRL-Robot-Arm
featured: true
legacyPath: /portfolio/5-deep-rl-arm/
---

- This project is based on the Nvidia open source project "jetson-reinforcement" developed by [Dustin Franklin](https://github.com/dusty-nv). The goal of the project is to create a DQN agent and define reward functions to teach a robotic arm to carry out two primary objectives:
  1. Have any part of the robot arm touch the object of interest, with at least a 90% accuracy.
  2. Have only the gripper base of the robot arm touch the object, with at least a 80% accuracy.
