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
draw_image = np.copy(image)

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

from scipy.ndimage.measurements import label

ystart = 400
ystop = 656
scales_d = {
    0.9:[400,656],
    1.5:[400,656],
    2.0:[400,656]
}
 
  
class Heatmaps():
    def _init_(self):
        self.maxlen = 0
        self.heatmaps = []
        
    def average_heat(self, heatmap, maxlen):
        self.maxlen = maxlen
        self.heatmaps.append(heatmap)

        print("this is the lenght of my heatmap",len(self.heatmaps))
        if len(self.heatmaps) == self.maxlen:        
            return sum(self.heatmaps)/self.maxlen
        else:
            return sum(self.heatmaps)/len(self.heatmaps)
        
        #return sum(self.heatmaps)
  
def pipeline(image):

    heat_d = {}
    heat_image_d = {}
    head_map_d = {}
    heatmap_d = {}

    out_img_d = {}
    bbox_list_d = {}

    for scale,y_coord in scales_d.items():  
        out_img_d[scale], bbox_list_d[scale] = find_cars(image, y_coord[0], y_coord[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)


    coord_li = []
    for key,box_li in bbox_list_d.items():
        for box in box_li:
            coord_li.append( ( ( box[0],box[1] ), ( box[2], box[3] ) ) ) 

    heat_image = np.zeros_like(image[:,:,0]).astype(np.float)


    heatmap1 = add_heat(heat_image, coord_li)
    heatmap2 = heat.average_heat(heatmap1, maxlen) 
    heatmap3 = apply_threshold(heatmap2,1.5)
    
    #heat_max = np.clip(heat, 0, 255)
       
    labels = label(heatmap3)
    draw_img4 = draw_labeled_bboxes(np.copy(image), labels)
    heat_lbl = draw_labeled_bboxes(np.copy(heatmap3), labels)
    #draw_img4 = show_image(draw_img4, heatmap3, (labels[1]) )

    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img4
    #return heatmap3

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque

  
vid_output = "testvideos/test1.mp4"
#clip1 = VideoFileClip("challenge_video.mp4").subclip(0,10)
def process_video(image):
    clip = pipeline(image)
    return clip

maxlen = 25     # number of frames to average
heat = Heatmaps()
heat.heatmaps = deque(maxlen = maxlen)

#clip1 = VideoFileClip("project_video.mp4")
clip1 = VideoFileClip("project_video.mp4").subclip(38,50)
white_clip = clip1.fl_image(process_video)
white_clip.write_videofile(vid_output, audio=False)
