# **Vehicle Detection Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  ![Language](https://img.shields.io/badge/language-Python-green.svg)


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/pipeline_test_images.png  "Pipeline over Test Images"
[image2]: ./output_images/sliding_window_search_test1.png "Sliding window searching result over test1.jpg"
[image3]: ./output_images/false_positive_rejection.png "False Positive Rejection with Thresholded Heatmap"

## Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/viewindividually) and describe how I addressed each point in my implementation.  

 ---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
I started by loading the the labeled data for vehicle and non-vehicle examples provided on the class instructions in cells `[3]` through `[5]` of the [Ipython notebook](). These images are collected into a numpy array `X` and 
and splitted into training and test data along with the label vector `y`.

I followed class instructions and created several functions in `DataProcessing/preprocess.py`.  The function names and their usage are listed in the following table:

| Function Name |  Usage |
|-------------|-------------|
|`convert_color()` |  convert image color space from BGR to another space |
|`bin_spatial()` | convert image to specified size and vectorize it |
| `color_hist()` |  a vector of histogram counts of pixel values in all three channels |
| `get_hog_features()` | histogram of gradient (HOG) features   |
| `extract_features()` | load images from a list of file paths, call all four functions above and return a matrix of features where rows are image files and columns are features  |
| class `Extract`  |  A class inherited from  `BaseEstimator` and `TransformerMixin`.  It transforms list/array of images to matrix of features |

During data processing, `extract_features()` or an instance of `Extract` is called to extract the spatial,
histogram of pixel-value, and histogram of gradient features.  

#### 2. Explain how you settled on your final choice of HOG parameters.

I did three-fold cross validation to search the parameters.
I used `RandomizedSearchCV` to uniformly search over the following parameter sets:
```python
param_grid = {'ext__color_space': ['HLS', 'YUV', 'YCrCb'], 'ext__spatial_size': [(24, 24), (32, 32), (40, 40)],
              'ext__hist_bins': [16, 32, 48], 'ext__orient': [6, 9, 12], 'ext__pix_per_cell': [6, 8, 10], 
              'ext__hog_channel': [0, 1, 2, 'ALL'], 'ext__cell_per_block': [2, 3],
              'svc__C': np.logspace(-4, 4, 9)}
```
`ext` here represents an instance of the `Extract` class
I defined in `DataProcessing/preprocess.py`. `svc` is an instance of `LinearSVC`, imported from `sklearn.svm`.
As is indicated in the documentation of scikit learn, 
`LinearSVC` is a more efficient implementation of support vector machine (SVM) classifier with linear kernel. 
In my other experimental trials, I did cross validation using SVM with radial basis function (RBF) kernel and 
random forest classifier. None of these classifiers outperform linear SVM, so here I am only using linear SVM.

The scoring criteria I used is `accuracy`. I also tried using `f1` and `roc_auc`.  From my eye ball inspection,
the former seems to be a more conservative approach, while `roc_auc` might be prone to overfitting.  I think `accuracy`
strikes a balance in this particular case.

I randomly sampled and searched over 30 parameter sets and perform three-fold cross validation over the training data `X_train` and `y_train`.  Below is the best set of parameters among the 30 samples:
```python
{'ext__cell_per_block': 2,
 'ext__color_space': 'YUV',
 'ext__hist_bins': 48,
 'ext__hog_channel': 'ALL',
 'ext__orient': 9,
 'ext__pix_per_cell': 8,
 'ext__spatial_size': (24, 24),
 'svc__C': 100.0}
```
Moving forward, I will use the above set of parameters to train my classifier. 

 #### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
The model training procedure can be find in cell `[26]｀.  Basically I create a pipeline that extracts features,
standardize the features, then fit a Linear SVM classifier to the features:
``` python
pipeline = Pipeline([('ext', Extract()), ('imp', Imputer()), ('scl', StandardScaler()), ('svc', LinearSVC())])
pipeline.fit(X, y)
```

 ### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I followed the approach demonstrated in the section of **Hog Sub-sampling Window Search** on the class website to implement the sliding window searching procedure.
This approach only compute HOG for the whole slice of image for each scaling factor.
My implementation of the scanning approach is defined in `DataProcessing/scan.py` and the main function called by the pipeline is `find_cars()`. The actual scanning is done in the function `car_scan()` and the function `draw_create_boxes()`
draws and creates the bounding boxes for detected regions.

Basically I only searched cars within the (vertical) bottom half of the image.  I scanned vertical slices of the image
with different scaling factors sequentially.  Below is the code for the search slices and their scaling factors:
```python
search_range = [(h//2, h//2 + slice_height) for slice_height in range(50, h//2, 50)] + [(h//2, h)]
search_scaling = [(end - start) / 250 * 1.5 for start, end in search_range] 
```
I started from searching over a 50-pixel slice, then over 100-pixel slice, and so forth till `h/2`-pixel slice where
`h` is the height of the image. As for the scaling factor, the scaling factor is proportional to the height of the slice
and I followed the class example to set the scaling factor be 1.5 for the 250-pixel slice (in class it was 256-pixel slice).  

As for the overlapping between windows, I didn't tune the overlapping size and I would step 2 cells forward right/down
for every iteration, which is 6 overlapping cells between the windows next to each other (horizontally/vertically).

Below is the result of sliding window searching result over `test1.jpg`.
![][image2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below is the result of the pipeline over the six test images.
![][image1]
I did several experiment to tune the classifier. First of all, as mentioned before, I tried tuning the feature extraction approach and SVM parameter with cross validation and different scoring criterion such as accuracy, roc_auc and f1.  I found accuracy seem to be a more suitable criteria then the rest two.  I also tuned the range of slice heights and
the threshold of heat map. But it is more of trial and error in both of these cases.  It seems that a threshold of 2
for the heatmap is more robust. This threshold means that a pixel will only be included in the final bounding box
if it lies in three or more scanning-detected bounding box.


## Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Below is a link to my project video results:

[![Project Video](http://img.youtube.com/vi/1E-Xtn-5sWM/0.jpg)](https://youtu.be/1E-Xtn-5sWM "Project Video")

you can also find it in this [repo](output_videos/project_video.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
The procedures for rejecting false positives are implemented　in `DataProcessing/pipeline.py`.  The basic idea follows
the class instructions, first a heatmap is created by counting the number of times individual pixels are found in
the bounding boxes found by the scanning approach. This followed by thresholding the heatmap at 2
and using `label()` imported from `scipy.ndimage.measurements` to find connected regions. These regions are drawn with
`draw_labeled_bboxes()`, here I also discard bounding boxes that are too small (diagonal distance < 50 pixels), 
as they are likely to be distant cars (which are less interesting) or false positives.

Below is a demonstration of the pipeline of forming heatmap, thresholding and forming connected regions and drawing final boxes:
![][image3]

During video processing I also smooth the heatmap by forming a weighted average of past heatmaps
and present one:
```python
heatmap = alpha*heatmap+ sum((1-alpha)**(l-1-i) * alpha**(i>0) * pipeline.hist[i] for i in range(l-1))
```
where `alpha = 0.5`.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are situations that my pipeline will detect trees as cars.  It seems to be a result of color and HOG being similar
to car at times.  Another case is that when the car is on the very horizontal boundary of the scene and only part of it
is seen. It seems that some kind of time series prediction procedure for tracking the boundary points of the bounding boxes may help.  Even some kind of extrapolation might help.  Also, when two cars are kind of next to each other in the
video clip, it can be difficult to separate them.  Finally, my procedure is by far not real time.

I think it might be helpful to use intersection over union to smooth bounding boxes across video frames.
With this, we could potentially group bounding boxes that belong to the same car and maybe interpolate or remove
misdetections.  Another useful procedure is maybe use some kind of tracking procedure to track the boundary point.
Finally, it might be more efficient to use modern procedures that utilize deep learning, such as YOLO (you only look once).













