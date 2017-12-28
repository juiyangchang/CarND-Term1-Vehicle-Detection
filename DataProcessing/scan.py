import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

import DataProcessing.preprocess as proc
from DataProcessing.preprocess import Extract

def car_scan(img, ystart, ystop, scale, color, imputer, classifier):
    spatial_size = (24, 24)
    hist_bins = 48
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    cells_per_step = 2  # Instead of overlap, define how many blocks to step
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64

    img_tosearch = img[ystart:ystop,:,:]
    
    # try 'HSV', 'LUV', 'HLS', 'YUV', or 'YCrCb'
    ctrans_tosearch = proc.convert_color(img_tosearch, color) 
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (int(imshape[1]/scale), int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block*cell_per_block
    
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1    # 64 // 8 - 2 + 1 = 7
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1 
    # say nxblocks = 14, (14 - 7) // 2 + 1 = 4, xb = 0, 1, 2, 3 and xpos = 0, 2, 4, 6
    # and x_pos+n_blocks_per_window = 7, 9, 11, 13
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = proc.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = proc.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = proc.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    ret = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = proc.bin_spatial(subimg, size=spatial_size)
            hist_features = proc.color_hist(subimg, nbins=hist_bins)

            X = np.hstack((spatial_features, hist_features, hog_features)).reshape((1,-1))
            X = imputer.transform(X)
            test_prediction = classifier.predict(X)
            
            if test_prediction == 1:
                ret.append((xleft, ytop))
            
            #if test_prediction == 1:
            #    xbox_left = np.int(xleft*scale)
            #    ytop_draw = np.int(ytop*scale)
            #    win_draw = np.int(window*scale)
            #    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return ret

def draw_create_boxes(draw_img, window_list, scale, ystart):
    window = 64
    bbox_list = []
    for xleft, ytop in window_list:
        xbox_left = np.int(xleft*scale)
        ybox_top = np.int(ytop*scale)+ystart
        win_draw = np.int(window*scale)
        
        bbox_list.append(((xbox_left, ybox_top), (xbox_left+win_draw, ybox_top+win_draw)))
        
        cv2.rectangle(draw_img, (xbox_left, ybox_top), 
                      (xbox_left+win_draw, ybox_top+win_draw), (255,0,0), 6) 
    return draw_img, bbox_list

def find_cars(img, X, y, color='RGB', C=1e-4):
    if not hasattr(find_cars, 'classifier') or find_cars.color != color or find_cars.C != C:
        ext = Extract(color_space=color, spatial_size=(24, 24), hist_bins=48, orient=9, pix_per_cell=8, 
            cell_per_block=2, hog_channel='ALL')
        imptr = Imputer()
        pipeline = Pipeline([('scl', StandardScaler()), ('svc', LinearSVC(C=C))])

        X = ext.fit_transform(X) 
        X = imptr.fit_transform(X)
        pipeline.fit(X, y)
        
        find_cars.imptr = imptr
        find_cars.classifier = pipeline   
        find_cars.color = color
        find_cars.C = C
        
    h, w = img.shape[:2]
    #search_range = [(400, 656)] 
    search_range = [(h//2, h//2 + rng_size) for rng_size in range(50, h//2, 100)] + [(h//2, h)]
    #search_scaling = [1.5] 
    search_scaling = [(end - start) / 250 * 1.5 for start, end in search_range] 
    
    windows = []    
    for i, search_window in enumerate(search_range):
        windows.append(car_scan(img, search_window[0], search_window[1], search_scaling[i], 
                                color, find_cars.imptr, find_cars.classifier))
        
    draw_img = np.copy(img)
    bboxes = []
    for i, window_list in enumerate(windows):
        draw_img, bbox_list = draw_create_boxes(draw_img, window_list, search_scaling[i], search_range[i][0])
        bboxes.extend(bbox_list)
    return draw_img, bboxes