import numpy as np
import cv2
from scipy.ndimage.measurements import label
import pickle
import DataProcessing.scan as scan
from collections import deque

data = pickle.load(open('data.p', 'rb'))
X = data['X']
y = data['y']

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    thresholded = np.copy(heatmap)
    thresholded[thresholded <= threshold] = 0
    # Return thresholded map
    return thresholded

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    draw_img = np.copy(img)
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        if np.sqrt((bbox[1][0] - bbox[0][0])**2 + (bbox[1][1] - bbox[0][1])**2) > 50:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return draw_img

def pipeline(img):
    alpha = 0.5
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    _, box_list = scan.find_cars(img, X, y, 'YUV', C=100)
    
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, box_list)

    thresholded = apply_threshold(heatmap, 1)
    if hasattr(pipeline, 'hist_thresholded'):
        pipeline.hist_thresholded.append(thresholded)
        if len(pipeline.hist_thresholded) > 5:
            pipeline.hist_thresholded.popleft()

        l = len(pipeline.hist_thresholded)
        thresholded = alpha*thresholded + sum((1-alpha)**(l-1-i) * alpha**(i>0) * pipeline.hist_thresholded[i] for i in range(l-1))
    else:
        pipeline.hist_thresholded = deque([thresholded])
    
    labels = label(thresholded)
    #print(labels[1], 'cars found')
    #plt.imshow(labels[0], cmap='gray')
    
    result = draw_labeled_bboxes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), labels)
    result = cv2.resize(result,None,fx=0.66, fy=0.66, interpolation = cv2.INTER_AREA)
    return result