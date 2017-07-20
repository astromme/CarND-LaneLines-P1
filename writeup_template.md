# **Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)
[original]: ./examples/0.base_image.png "Original Image"
[hsv_value]: ./examples/1.hsv_value.png "HSV Value"
[threshold]: ./examples/2.threshold.png "Threshold"
[gaussian_blur]: ./examples/3.gaussian_blur.png "Gaussian Blur"
[canny]: ./examples/4.canny.png "Canny"
[roi_inside]: ./examples/5.image_roi_inside.png "ROI Inside"
[roi_outside]: ./examples/6.image_roi_outside.png "ROI Outside"
[hough_lines]: ./examples/7.hough_lines.png "Hough Lines"
[final]: ./examples/8.final_image.png "Final Image"

---

### 1. Description of my pipeline

My general strategy is to use as much domain knowledge as possible (e.g. that the camera is in a fixed position relative to the car and that lane lines are bright compared to the road surface) to eliminate background noise, then perform line detection, then simplify those into a left and right lane line.

My pipeline consists of 7 steps. The first 6 are more or less direct cv2 commands, and the final step is my logic.

![original image][original]

#### a. HSV Value
The image is converted to hue-saturation-value (HSV) color space and the value component is extracted. This helps with the bridge portion of the challenge image and performs better than grayscale.

![hsv value][hsv_value]

#### b. Truncation
A truncation threshold for values 175-255 is applied, done by first inverting the image, then applying from 0-80, then inverting again. This is because cv2.THRESH_TRUNC has no inverse version, and cv2.THRESH_TRUNC is needed because the simple binary threshold creates many more sharp changes that canny detects as edges.

```python
pipeline.add('threshold', 255 - cv2.threshold(255 - pipeline.image(),80,255,cv2.THRESH_TRUNC)[1])    
```

![threshold][threshold]

#### c. Gaussian Blur
Gaussian blur is applied with a kernel size of 5 to remove high frequency noise which helps the canny edge detection perform better.

```python
pipeline.add('gaussian_blur', gaussian_blur(pipeline.image(), 5))
```

![gaussian_blur][gaussian_blur]

#### d. Canny Edge Detection

Canny edge detection is applied. I chose 50 and 150 as thresholds after tying other combinations like 100, 150 and 80, 200.

```python
pipeline.add('canny', canny(pipeline.image(), 50, 150))
```

![canny][canny]

#### e. Region of Interest (keep the inside)
A region of interest is applied that eliminates most of the image outside the lane. It is computed dynamically based on the size of the image. I chose values such that it worked reasonably well for all three videos.

```python
topleft = (int(0.46*width), int(0.63*height))
topright = (int(0.54*width), int(0.63*height))
bottomleft = (int(0.18*width), int(0.9*height))
bottomright = (int(0.88*width), int(0.9*height))    

vertices = np.array([[topleft, topright, bottomright, bottomleft]], dtype=np.int32)

pipeline.add('image_roi_inside', region_of_interest(pipeline.image(), vertices))
```

![roi_inside][roi_inside]


#### f. Region of Interest (keep the outside)
A second region of interest is applied that eliminates most of the image inside the lane. The vertices are slightly moved versions of the roi_inside vertices.

```python
outside_vertices = np.array([[
    [topleft[0] + 0.3 * top_width, topleft[1]],
    [topright[0] - 0.3 * top_width, topright[1]],
    [bottomright[0] - 0.2 * bottom_width, bottomright[1]],
    [bottomleft[0] + 0.2 * bottom_width, bottomleft[1]],
]], dtype=np.int32)
pipeline.add('image_roi_outside', region_of_interest(pipeline.image(), outside_vertices, keep_outside=True))
```


![roi_outside][roi_outside]


#### g. Hough Lines

Finally, hough lines are computed, and a single line is fit for each side of the lane. This is done in a few parts.

First, HoughLinesP is used to get the hough lines.

```python
lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
```

Second, lines are split into left and right roughly down the center.
```python
left_lines, right_lines = split_lines(lines)
```

Third, lines with slopes that are too horizontal are eliminated.
```python
left_lines = [line for line in left_lines if slope(line) < -0.4]
right_lines = [line for line in right_lines if slope(line) > 0.4]
```

Fourth, least squares is used to fit the best line for each side.
```python
leftM, leftC = best_fit_line_least_squares(left_lines)
rightM, rightC = best_fit_line_least_squares(right_lines)
```

Fifth, the lines are averaged with the lines from the previous step. This can be done because of domain knowledge that the lines are unlikely to change much from one frame to the next.

```python
# average the lines with the previous frame's lines
leftM = (leftM + p_leftM) / 2
rightM = (rightM + p_rightM) / 2
leftC = (leftC + p_leftC) / 2
rightC = (rightC + p_rightC) / 2
```

![hough_lines][hough_lines]

#### f. Final Image

Notice how even in this complex image with bridge & shadow the lines are correctly computed.

![final_image][final]

### 2. Potential shortcomings with my current pipeline

 - The region of interest is fragile. It will break if the camera is turned left or right a bit. It will also break when driving in a lane with a nonzero second derivative (a.k.a. the bottom of a valley or top of a hill).
 - The pipeline won't handle different lighting conditions well because of the hardcoded threshold values.
 - I'm not sure what will happen when a car merges into a lane. It seems likely that there will be many more candidate lines, and this could cause the least squares computation to have an incorrect computed line. The ROI cropping should help with cars that are in the same lane.
 - The computed lines have jitter. The smoothing helps somewhat, but it could still be better.

### 3. Possible improvements to my pipeline

* Improving the detection to not require a region of interest could make it more robust to situations that break the region of interest. Perhaps this could be done by limiting slope more rigorously. Another approach would be to better select which hough lines are correct, perhaps by observing how many segments are present for a given line.
* Setting thresholding values adaptively could help with different lighting conditions.
* Weighted least squares could help the algorithm correctly consider long lines to be more important than short dots. Right now every segment is considered equally.
* Identifying key points between frames could help localize and reduce jitter. One method would be to identify line segments of similar length, slope, and position between frames.
