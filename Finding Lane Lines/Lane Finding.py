####################################################################################################
####################################################################################################
# Name: Color Selection
# Coder: Janson Fong
# Description: This script return road lanes of an image by setting the background black
#
####################################################################################################

####################################################################################################
# Libraries and Modules
####################################################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
####################################################################################################
# Constants
####################################################################################################
IMAGE_PATH = r'../Images/test.jpg'
RED_THRESHOLD = 200
GREEN_THRESHOLD = 100
BLUE_THRESHOLD = 200
####################################################################################################
# Method Definitions
####################################################################################################

####################################################################################################
# Class Definitions
####################################################################################################

####################################################################################################
# Main Program
####################################################################################################

# Read lane image and display basic statistics 
image = mpimg.imread(IMAGE_PATH)
print('This image is: ', type(image), 'with dimensions:', image.shape)

# Make copy of image 
ySize = image.shape[0]
xSize = image.shape[1]
colorSelect = np.copy(image)

# Loop through each pixel and compare it with threshold value
for x in range(xSize):
	for y in range(ySize):
		red = image[y, x, 0]
		green = image[y, x, 1]
		blue = image[y, x, 2]

		if (red < RED_THRESHOLD or green < GREEN_THRESHOLD or blue < BLUE_THRESHOLD):
			colorSelect[y, x, :] = [0, 0, 0]

plt.imshow(colorSelect)
plt.show()