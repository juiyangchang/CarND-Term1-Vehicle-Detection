# **Vehicle Detection Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  ![Language](https://img.shields.io/badge/language-Python-green.svg)


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/viewindividually) and describe how I addressed each point in my implementation.  

 ---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
I started by loading the the labeled data for vehicle and non-vehicle examples provided on the class instructions in cells `[3]` through `[5]` of the [Ipython notebook](). These images are collected into a numpy array `X` and 
and splitted into training and test data along with the label vector `y`.

I followed class instructions and created several functions in `DataProcessing/preprocess.py`.  The functions are listed in the following table:

| Function Name |  Usage |
|-------------|-------------|
|`convert_color()` |  convert image color space from BGR to another space |
|`bin_spatial()` | convert image to specified size and vectorize it |
| `color_hist()` |   histogram of pixel values in all three channels |
| `get_hog_features()` | histogram of gradient features   |
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
`LinearSVC` is an more efficient implementation of support vector machine (SVM) classifier with linear kernel.  In the initial search, the `cell_per_block`

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
Moving forward, I will use the above set of parameters to train my classifer. 

 #### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
The model training procedure can be find in cell `[26]｀.  Basically I create a pipeline that extracts features,
standardize the features, then fit to the features with Linear SVM classifier:
``` python
pipeline = Pipeline([('ext', Extract()), ('imp', Imputer()), ('scl', StandardScaler()), ('svc', LinearSVC())])
pipeline.fit(X, y)
```

 ### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I followed the approach demonstrated on the section of **Hog Sub-sampling Window Search** on the class website to implement the sliding window search.  
This approach only compute HoG for the whole slice of image for each scaling factor.
My implementation of the scanning approach is defined in `DataProcessing/scan.py` and the main function called by the pipeline is `find_cars()` .  The actual scanning is done in the function `car_scan()` and the function 

Basically I only searched cars within the (vertical) bottom half of the image.  I scanned vertical slices of the image
with different scaling factors sequentially.  Below is the code:
```python
search_range = [(h//2, h//2 + slice_height) for slice_height in range(50, h//2, 50)] + [(h//2, h)]
search_scaling = [(end - start) / 250 * 1.5 for start, end in search_range] 
```
I started from searching over a 50-pixel slice, then over 100-pixel slice, and so forth till `h/2`-pixel slice where
`h` is the height of the image. As for the scaling factor, the scaling factor is proportional to the height of the slice
and I followed the class example to set the scaling factor be 1.5 for the 250-pixel slice (in class it was 256-pixel slice).  

As for the overlapping between windows, I didn't tune the overlapping size and I would step 2 cells forward right/down
for every iteration, which is 6 overlapping cells between the windows next to each other (horizontally/vertically).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?



## Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.











