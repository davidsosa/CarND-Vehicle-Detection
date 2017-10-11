import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import time
import pickle

from functions import *

# load dictionary with classificator
clf_dic = pickle.load( open( "clf_dic_Ycolor_7000.p", "rb" ) )
svc = clf_dic["svc"]  
X_scaler = clf_dic["X_scaler"]  
### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()

image = mpimg.imread('test_images/test1.jpg')
image = image.astype(np.float32)/255
draw_image = np.copy(image)

ystart = 400
ystop = 656

scales = [0.9,1.5,2.0 ]

hot_images = []
all_images = []

for scale in scales:

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(int(scale*64), int(scale*64)), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, scale=scale, 
                                 spatial_size=spatial_size, hist_bins=hist_bins, 
                                 orient=orient, pix_per_cell=pix_per_cell, 
                                 cell_per_block=cell_per_block, 
                                 hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                 hist_feat=hist_feat, hog_feat=hog_feat)                       

    all_window_img = draw_boxes(draw_image, windows, color=(0, 0, 1), thick=6)                    
    hot_window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 1), thick=6)                    

    hot_images.append(hot_window_img)
    all_images.append(all_window_img)

fig = plt.figure()

counter = 1
counter1 = 1
for scale in scales:
    plt.subplot("32"+str(counter))
    plt.imshow( all_images[counter1-1])
    plt.title('All windows (overlap 0.5) Scale: '+str(scale))
    counter +=1 
    plt.subplot("32"+str(counter))
    plt.imshow(hot_images[counter1-1])
    plt.title('Hot windows')
    fig.tight_layout()

    counter +=1
    counter1 +=1
  
#plt.show()
plt.savefig("./myimages/hotwindows.jpg")


scale_dic = {}

#image = mpimg.imread('testvideos/output_images/frame203.jpg')
#image = mpimg.imread('testvideos/output_images/frame190.jpg')
#image = mpimg.imread('testvideos/output_images/frame220.jpg')

#image = image.astype(np.float32)/255

image = mpimg.imread('test_images/test1.jpg')
draw_image = np.copy(image)

scales_d = {
    0.90:[400,656,0,1280],
    1.5:[400,656,0,1280],
    2.0:[400,656,0,1280],
}

out_img_d = {}
bbox_list_d = {}
for scale,coords in scales_d.items():  
    out_img_d[scale], bbox_list_d[scale] = find_cars(image, coords[0], coords[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

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
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

from scipy.ndimage.measurements import label

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars

    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y                  
        bbox = ((np.min(nonzerox),np.min(nonzeroy)),(np.max(nonzerox), np.max(nonzeroy) ))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

coord_li = []
for key,box_li in bbox_list_d.items():
    for box in box_li:
        coord_li.append( ( ( box[0],box[1] ), ( box[2], box[3] ) ) ) 
    
window_img = draw_boxes(draw_image, coord_li, color=(0, 0, 255), thick=6)                    
#plt.imshow(window_img)
#plt.show()

heat_image = np.zeros_like(image[:,:,0]).astype(np.float)
heat = add_heat(heat_image, coord_li)
heat = apply_threshold(heat,1)
heatmap = np.clip(heat, 0, 255)

### Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(221)
plt.imshow(window_img)
plt.title('All hot windows')
plt.subplot(222)
plt.imshow(heatmap, cmap='hot')
plt.title('Heatmap')
plt.subplot(223)
plt.imshow(draw_img)
plt.title('Labeled Bounding boxes')
fig.tight_layout()
#plt.savefig("./myimages/hotwindows_heatmap.jpg")
plt.show()
