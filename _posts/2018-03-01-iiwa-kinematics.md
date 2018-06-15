---
title: 'IIWA Kinematics'
date: 2018-03-01
permalink: /posts/2018/iiwa-kinematics
tags:
  - ROS
  - kuka iiwa
  - inverse kinematics
---

Given a list of end-effector poses, calculate joint angles using Inverse Kinematics for the KUKA LBR IIWA R820

Kinematics implementation for the KUKA LBR IIWA R820 (14 Kg). 

Ubuntu 16.04 + ROS Kinetic

Kinematics implementation for the KUKA LBR IIWA R820 (14 Kg).
![alt text][gif]
Video: https://youtu.be/L5daeWuy1js

[//]: # "Image References"
[gif]: /images/portfolio/pick-place/pick-place.gif
[fk]: /images/portfolio/pick-place/imgs/forward_kinematics.jpg
[results]:/images/portfolio/pick-place/imgs/IK_results.jpg
[dh]:/images/portfolio/pick-place/imgs/dh.jpg
[ik_1]:/images/portfolio/pick-place/imgs/ik_1.jpg



### Getting Started

If you do not have an active ROS workspace, you can create one by:

```
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```

Clone this repo into the **src** directory of your workspace:

```
$ cd ~/catkin_ws/src
$ git clone https://github.com/gwwang16/iiwa_kinematics.git
```

Install dependencies

```
$ cd ~/catkin_ws
$ sudo apt-get update
$ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
$ cd ~/catkin_ws/src/iiwa_kinematics/iiwa_arm/scripts
$ chmod +x safe_spawner.sh
$ chmod +x target_spawn.py
$ chmod +x IK_server.py
```

Build the project:

```
$ cd ~/catkin_ws
$ catkin_make
```
Add following to your .bashrc file
```
$ export GAZEBO_MODEL_PATH=~/catkin_ws/src/iiwa_kinematics/iiwa_arm/models
$ source ~/catkin_ws/devel/setup.bash
```


### Forward kinematics demo

For demo mode make sure the demo flag is set to "true" in `inverse_kinematics.launch` file under iiwa_kinematics/iiwa_arm/launch

```
$ roslaunch iiwa_arm forward_kinematics.launch
```
![alt text][fk]


### Launch the project

```
$ cd ~/catkin_ws/src/iiwa_kinematics/iiwa_arm/scripts
$ ./safe_spawner.sh
```

To run your own Inverse Kinematics code change the demo flag described above to "false" and run your code (once the project has successfully loaded) by:

```
$ cd ~/catkin_ws/src/iiwa_kinematics/iiwa_arm/scripts
$ rosrun iiwa_arm IK_server.py
```
![alt text][results]



### IK Steps

![alt text][dh]

Table.I The relative location of joint i-1  to  i

| Joint   | X    | Y      | Z      | Roll, Pitch, Yaw |
| ------- | ---- | ------ | ------ | ---------------- |
| 1       | 0    | 0      | 0.1575 | `(0, 0, 0)`      |
| 2       | 0    | 0      | 0.2025 | `(pi/2, 0, pi)`  |
| 3       | 0    | 0.2045 | 0      | `(pi/2, 0, pi)`  |
| 4       | 0    | 0      | 0.2155 | `(pi/2, 0, 0)`   |
| 5       | 0    | 0.1845 | 0      | `(-pi/2, pi, 0)` |
| 6       | 0    | 0      | 0.2155 | `(pi/2, 0, 0)`   |
| 7       | 0    | 0.081  | 0      | `(-pi/2, pi, 0)` |
| gripper | 0    | 0      | 0.08   | `(0, -pi/2, 0)`  |

Now, we can obtain our modified DH table. 

**Note:** joint3 is fixed to remove redundant dof.

Table. II The modified DH parameters

| Joint | $\alpha_{i-1}$ | $a_{i-1}$ | $d_i$   | $\theta_i$    |
| ----- | -------------- | --------- | ------- | ------------- |
| 1     | 0              | 0         | `0.36`  | `q1`          |
| 2     | `-pi/2`        | 0         | 0       | `q2:q2-pi/2`  |
| 4     | 0              | `0.42`    | 0       | `q4:-q4+pi/2` |
| 5     | `-pi/2`        | 0         | `0.4`   | `q5`          |
| 6     | `pi/2`         | 0         | 0       | `q6`          |
| 7     | `-pi/2`        | 0         | 0       | `q7`          |
| g     | 0              | 0         | `0.161` | `q8:0`        |

#### theta1,2,4

**Note:** `R_corr` is 1 here.  Because reference frame between URDF O7 and DH O7 are same, IK server called  Link 7 pose immediately for convenience.

![alt text][ik_1]

#### theta5-7


```
R4_7 = 
Matrix([
[-sin(q5)*sin(q7) + cos(q5)*cos(q6)*cos(q7), -sin(q5)*cos(q7) - sin(q7)*cos(q5)*cos(q6), sin(q6)*cos(q5)],
[                           sin(q6)*cos(q7),                           -sin(q6)*sin(q7),        -cos(q6)],
[ sin(q5)*cos(q6)*cos(q7) + sin(q7)*cos(q5), -sin(q5)*sin(q7)*cos(q6) + cos(q5)*cos(q7), sin(q5)*sin(q6)]])
```


```
def Euler_angles_from_matrix_URDF(R, angles_pre):
    r13 = R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2] 
    r32, r33 = R[2,1], R[2,2]
    if r23 is not 0:
        q6 = atan2(sqrt(r13**2 + r33**2), -r23)
        if sin(q6) < 0:
            q5 = atan2(-r33, -r13)
            q7 = atan2(r22, -r21)
    	else:
            q5 = atan2(r33, r13)
            q7 = atan2(-r22, r21)
    else:
        q5 = angles_pre[4]
        q6 = angles_pre[5]
        q7 = angles_pre[6]
return np.float64(q5), np.float64(q6), np.float64(q7)
```

---
References:

Udacity kinematics project:
https://github.com/udacity/RoboND-Kinematics-Project

iiwa urdf and gazebo package:
https://github.com/rtkg/lbr_iiwa

Computing Euler angles from a rottion matrix
http://thomasbeatty.com/MATH%20PAGES/ARCHIVES%20-%20NOTES/Applied%20Math/euler%20angles.pdf