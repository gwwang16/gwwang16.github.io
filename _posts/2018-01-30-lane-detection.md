---
title: 'Lane Detection'
date: 2018-01-29
permalink: /posts/2018/lane-detection
tags:
  - deep learning
  - autonomous driving
---

Build a pipeline using distortion correction, image rectification, color transforms, and gradient thresholding to identify lane lines and their curvature.


# Advanced Lane Detection

Github repo https://github.com/gwwang16/CarND-Advanced-Lane-Lines




### Features:

* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: /images/portfolio/lane-detection/undistorted.png "Undistorted"
[image2]: /images/portfolio/lane-detection/undistorted_test1.png "Road Undistorted"
[image7]:  /images/portfolio/lane-detection/pespective.png "Pespective"
[image3]:  /images/portfolio/lane-detection/binary_result.png "Binary "
[image4]:  /images/portfolio/lane-detection/warped.png "Warp"
[image8]:  /images/portfolio/lane-detection/histogram.png "Historgram"
[image5]:  /images/portfolio/lane-detection/line_fitting.png "Fit Visual"
[image6]:  /images/portfolio/lane-detection/result.png "Output"
[gif]:  /images/portfolio/lane-detection/output.gif "resultt"
[image9]:  /images/portfolio/lane-detection/video_screenshot.png "Video output"

## Camera Calibration

### 1. Camera matrix and distortion coefficients.

The code for camera calibration is contained in `Camera_calibration.ipynb`

The steps contain:

- obtain `object points` and `corners` from chessboard pictures using `cv2.findChessboardCorners()`

- compute camera matrix and distortion coefficients using `cv2.calibrateCamera()` 

- apply distortion correction to image using `cv2.undistort()`

The following result is obtained

![alt text][image1]

## Pipeline (single images)

There are two main python files: `ImageProcess.py` and `LaneFinding.py`.

 `ImageProcess.py` is used for image processing contains distortion correction, extracting lane line pixels and image perspective.

`LaneFinding.py` is used for lane lines fitting, curvature calculating and draw lane, etc.

apply the distortion correction to one of the test images
![alt text][image2]


### 2. Thresholded binary image. 

I used a combination of color and sobel magnitude thresholds to generate a binary image.  I also tried gradient threshold and absolute sobel methods, but didn't find good result.  The color threshold method adopted the code was presented in slack channel by @kylesf. This code can be found in `combine_thresh()` in `ImageProcess.py`

Here's an example of my output for this step

![alt text][image3]

### 3. Perspective transform.

The code for perspective transform is `perspective()` in the file `ImageProcess.py` , it uses transform matrix, which is obtained during camera calibration using `cv2.getPerspectiveTransform(src, dst)`

The following source and destination points are used

|  Source   | Destination |
| :-------: | :---------: |
| 200, 720  |  350, 720   |
| 563, 470  |   350, 0    |
| 723, 470  |   980, 0    |
| 1130, 720 |  980, 720   |

I verified this perspective transform on a straight line image, the lines is parallel on the warped image, it means the perspective transform is reasonable.

![alt text][image7]

Then, the perspective transform is used into previous test image, the following result is obtained
![alt text][image4]



### 4. Identified lane-line pixels and fit their positions with a polynomial

We need locate the lane lines at first, the code is `find_lines_initial()` in `LaneFinding.py`.

 I split the image into 12 slices,  then

- determine left and right lines starting point of the bottom slice based on histogram result.

![alt text][image8]


- set a sliding window with `2*margin` pixels width and  `np.int(warped.shape[0]/nwindows)`  height around left and right starting points.

- set left and right lines starting points of the next slice as mean value of nonzero pixel indexes in the previous window

- repeat 2-3 steps to the last slice

- fit lane lines with 2 order polynomial using the above points of left and right, respectively.

  The result is the following 

  ![alt text][image5]



Once we know where the lines are, the sliding windows step can be skipped, the function `find_lines()` is used here. The sliding window is replaced by the adjacent domains around the previous fitting lines.



### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated by `curvature()` and the lane offset is calculated by `lane_offset()` in `LaneFinding.py`

#### 1) Curvature

We have obtained the fitting polynomial 

$$f(y) = Ay^2 + By + C$$

>
> fitting for f(y), rather than f(x), because the lane lines in the warped image are near vertical and may have the same x value for more than one y value.
>

The radius of curvature at any point x of the function x=f(y) is given as follows:

$$R_{curve} = (1+(2Ay+B)^2)^{(3/2)} / |2A|$$

The y values of image increase from top to bottom, so if we wanted to measure the radius of curvature closest to vehicle, we should evaluate the formula above at the y value corresponding to the bottom of  image.

Then, we should transfer the curvature from pixels into meters by multiplying coefficients, which can be calculated by measuring out the physical lane in the field of view of the camera. But rough estimated coefficients are used in this project

```
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

#### 2) Lane offset

The position of the vehicle with respect to center can be calculated with the relative position between image midpoint and lane midpoint. The code is the following


```
x_left = left_fit[0] * self.y_max**2 + left_fit[1] * self.y_max + left_fit[2]
x_right = right_fit[0] * self.y_max**2 + right_fit[1] * self.y_max + right_fit[2]
offset = (x_right - x_left) / 2. + x_left - self.midpoint
lane_width = x_right - x_left
# Transfer pixel into meter
offset = offset * self.xm_per_pix
```

### 6. Plotthe identified lane area  back down onto the road.

I used `draw_lane()` in file `LaneFinding.py` to draw the identified lane on the original image. 

Here is an example on a test image:
![alt text][image6]

---

## Result

Apply the proposed pipeline to video, the following result can be obtained.

![alt text][gif]

---

## Discussion

I implement a lane finding algorithm in this project, it performs well on the project video, but not well on the challenge video and fails on the hard challenge video. There are many points need to be improved further, but I have no time for this course, have to move into the next step. I hope I could improve it further in the future.

- The lane detecting algorithm is not robust enough for noise, such as shadow and blur lane lines, especially for the trees in the harder challenge video. The image filtering method need to be improved.
- The lane lines should be parallel for most of time, judging statement is preferred to select one better fitting line.