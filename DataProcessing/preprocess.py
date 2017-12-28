import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
from sklearn.base import TransformerMixin, BaseEstimator

def convert_color(img, conv='YCrCb'):
    conv_space = eval('cv2.COLOR_BGR2' + conv)
    return cv2.cvtColor(img, conv_space)
     
def bin_spatial(img, size=(32, 32)):
    # resize the image as 32-by-32 and 
    # make the pixel values in all channels and locations as a vector
    return cv2.resize(img, dsize=size).ravel()
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)# normed=True)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)# normed=True)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)# normed=True)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)

        # apply color conversion if other than 'RGB'
        # try 'HSV', 'LUV', 'HLS', 'YUV', or 'YCrCb'
        feature_image = convert_color(image, color_space) 

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            #feature_image = cv2.resize(feature_image, spatial_size)
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

class Extract(TransformerMixin, BaseEstimator):
    def __init__(self, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        
    def __repr__(self):
        return 'Extract(color_space={}, spatial_size={}, hist_bins={}, orient={}, pix_per_cell={}, cell_per_block={}, hog_channel={})'\
            .format(self.color_space, self.spatial_size, self.hist_bins, self.orient, self.pix_per_cell,
                    self.cell_per_block, self.hog_channel)
    
    def fit(self):
        pass
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)
        
    def transform(self, X, y = None):
        color_space = self.color_space
        spatial_size = self.spatial_size
        hist_bins = self.hist_bins
        orient = self.orient
        pix_per_cell = self.pix_per_cell
        cell_per_block = self.cell_per_block
        hog_channel = self.hog_channel
        
        X_new = []
        # Iterate through the list of images
        for x_i in X:
            # apply color conversion if other than 'RGB'
            # try 'HSV', 'LUV', 'HLS', 'YUV', or 'YCrCb'
            feature_image = convert_color(x_i, color_space) 
            
            spatial_features = bin_spatial(feature_image, size=spatial_size)

            hist_features = color_hist(feature_image, nbins=hist_bins)
        
            # Call get_hog_features() with vis=False, feature_vec=True
            #feature_image = cv2.resize(feature_image, spatial_size)
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                                         orient, pix_per_cell, cell_per_block, 
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                                                     pix_per_cell, cell_per_block, vis=False, 
                                                     feature_vec=True)
                
            # Append the new feature vector to the features list
            X_new.append(np.concatenate((spatial_features, hist_features, hog_features)))
        # Return list of feature vectors
        return np.vstack(X_new)