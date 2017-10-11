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

# Read in our vehicles and non-vehicles
images = glob.glob('*vehicles*/*/*.png')
cars = []
notcars = []

for image in images:
    if 'non' in image :
        notcars.append(image)
    else:
        cars.append(image)

print(len(notcars))
print(len(cars))

sample_size = 2500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

ind = np.random.randint(0, len(cars))
# Read in the image
imagecar = mpimg.imread(cars[ind])
imagenotcar = mpimg.imread(notcars[ind])

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(imagecar, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(imagenotcar, cmap='gray')
plt.title('Example Not a Car')
#plt.show()
plt.savefig("./myimages/car_notcar.jpg",bbox_inches='tight')


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
        cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
        cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features

## Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
   # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

## Define a function to compute color histogram features  
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features, channel1_hist, channel2_hist, channel3_hist

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
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

## Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
# Read in the image
image = mpimg.imread(cars[ind])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

features,hist1,hist2,hist3 = color_hist(image)

notcar_image = mpimg.imread(notcars[ind])
notcar_gray = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2GRAY)

# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

features_notcar, hog_notcarimage = get_hog_features(notcar_gray, orient, 
                                             pix_per_cell, cell_per_block, 
                                             vis=True, feature_vec=False)

# Plot the examples
fig = plt.figure()

plt.subplot(241)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image',fontsize=7)
plt.subplot(242)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization',fontsize=7)

plt.subplot(243)
plt.imshow(notcar_image, cmap='gray')
plt.title('Example Not Car Image',fontsize=7)
plt.subplot(244)
plt.imshow(hog_notcarimage, cmap='gray')
plt.title('HOG Visualization',fontsize=7)
plt.show()

plt.subplot(245)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image',fontsize=7)
plt.subplot(242)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization',fontsize=7)




plt.show()

#plt.savefig("./myimages/hogexample.jpg")
