## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.

* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./myimages/car_notcar.jpg
[image2]: ./myimages/hog_carsnon.jpeg
[image3]: ./myimages/hotwindows.jpg
[image6]: ./examples/labels_map.png
[image7]: ./myimages/hotwindows_heatmap.jpg
[video1]: ./testvideos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I defined the function `get_hog_features` in lines 15 through 26 of the file called `functions.py`. Then these features were extracted for the
vehicle and non-vehicle images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

The the HOG for both the vehicle and non-vehicle examples are shown in the following figure:  

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
The final choice of parameters was the default.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 7000 images (due to memory issues) with GTI vehicle image database. This can be seen in the script `trainer.py`. The SVN was trained both with the color histogram features in the `YCrCb` and the the HOG features in ALL the channels of the `YCrCb` color space.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The search region was restricted to the lower part of the image in the range of 400 to 656 in order to speed things up. I choose three scales (0.9,1.5 and 2.0) because I found this gave me the best performance. In the following figure the 

![alt text][image3]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The images can be seen above. The most relevant change implemented was  raining with the `YCrCb` color space instead of th `RGB` color space. This made the boxes much more stable around the cars and practically eliminated the false positives. 

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://www.youtube.com/watch?v=ck2mWLi3k6M)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The only difference between the pipeline for an image (implemented in `tracking.py`) and for a video (implemented in `pipeline.py`) is that heatmap class is created () to which averages the frames (in this case over 30 frames) and applies a slightly higher threshold of 1.5 (instead of 1.0). Other than that, the aplication of `scipy.ndimage.measurements.label()` is the same. With the label it is possible to construct the boxes around the car and the result can be observed in the video.

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One thing I struggled a lot with was that my classifier identifying the cars properly. I tried to play with the threshold and the number of frames averaged for the heatmap but nothing worked. Either the car was identified and we had too many false positives or the threshold was to high and the system would lose track of the car. The simple change of in training improve greatly the boxes around the cars making them more stable and reducing the false positives to a minimu. There is a lot of potential for improvement: for example I only trained with 7000 images with the GTI image database. In more complicated situations the amount of data could be a crucial difference for a succesful vehicle tracking.

Another topic that could use improvement as well is the parameter optmization. I only use the defaul parameters. The tools are the to tune the SVN parameters for a more optimal performance.     

