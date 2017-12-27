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
|------|---------|
|`convert_color()` |  convert image color space from BGR to another space |
|`bin_spatial()` | convert image to specified size and vectorize it |
| `color_hist()` |   histogram of pixel values in all three channels |
| `get_hog_features()` | histogram of gradient features   |
| `extract_features()` | load images from a list of file paths, call all four functions above and return a matrix of features where rows are image files and columns are features  |
| class `Extract`  |  A class inherited from  `BaseEstimator` and `TransformerMixin`.  It transforms list/array of images to matrix of features |

During data processing, `extract_features()` or an instance of `Extract` is called to extract the spatial,
histogram of pixel-value, and histogram of gradient features.  

#### 2. Explain how you settled on your final choice of HOG parameters.

I did two rounds of three-fold cross validation to search the parameters.
Initially I used `RandomizedSearchCV` to uniformly search over the following parameter sets:
```python
param_grid = {'ext__color_space': ['HLS', 'YUV', 'YCrCb'], 'ext__spatial_size': [(16, 16), (32, 32), (64, 64)],
              'ext__hist_bins': [16, 32, 64], 'ext__orient': [9, 12, 15], 'ext__pix_per_cell': [4, 8, 12], 
              'ext__hog_channel': [0, 1, 2, 'ALL'],               
              'svc__C': np.logspace(-4, 4, 9)}
```
`ext` here represents an instance of the `Extract` class
I defined in `DataProcessing/preprocess.py`. `svc` is an instance of `LinearSVC`, imported from `sklearn.svm`.
As is indicated in the documentation of scikit learn, 
`LinearSVC` is an more efficient implementation of support vector machine (SVM) classifier with linear kernel.  In the initial search, the `cell_per_block`

In my initial trial, I did cross validation using SVM with radial basis function (RBF) kernel and 
random forest classifier. None of these classifiers outperform linear SVM, so here I am only using linear SVM.

The scoring criteria I used is `roc_auc`. As was discussed on stackexchange, `roc_auc` represents [the expected proportion of positives ranked before a uniformly drawn random negative](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it). Hence maximizing `roc_auc` ensures the positive instances are more likely to be ranked above the negative instances. In retrospect, this might not be a good criterion for SVM as it may minimize the margin between positive and negative classes and make the classifier prone to overfitting.  

I randomly sampled and searched over 30 parameter sets and perform three-fold cross validation over the training data `X_train` and `y_train`.  Below is the set of parameters 
```python
{'ext__color_space': 'YUV',
 'ext__hist_bins': 32,
 'ext__hog_channel': 'ALL',
 'ext__orient': 12,
 'ext__pix_per_cell': 8,
 'ext__spatial_size': (16, 16),
 'svc__C': 0.0001}
```











