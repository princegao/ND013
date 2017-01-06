####################################################################################################
####################################################################################################
# Name: Lane Finding
# Coder: Janson Fong
# Description: 
#
####################################################################################################

####################################################################################################
# Libraries and Modules
####################################################################################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import math
####################################################################################################
# Constants
####################################################################################################

####################################################################################################
# Class Definitions
####################################################################################################

#
# The SectionApproximation class models a section as a straight line (y = mx + b). 
#
# Instantiate:
# 	To create an instance of this class, specify the start and end point of a section.
#
# Syntax:
#	SectionApproximation(P1, P2)
#
# Parameters:
#	Input Type: List[]
#	P1 = [X1, Y1]
#	P2 = [X2, Y2]
#
# Attributes:
#
# 	sectionSlope(): Returns the slope of section (m)
#	Return Type: Float
#	Return: m
#
#  	sectionIntercept(): Returns the intercept of a section (b)
#	Return Type: Float
#	Return: b
#
#	distancePointToSection(P0): Returns the distance between point P0 and the defined section 
#	Input Type: List[]
#	P0 = [X0, Y0]
#	Return Type: Float
#	Return: distance
#

class SectionApproximation:
	def __init__(self, P1, P2):
		self.X1 = P1[0]
		self.Y1 = P1[1]
		self.X2 = P2[0]
		self.Y2 = P2[1]

	def sectionSlope(self):
		if (self.X2 - self.X1 != 0):
			self.m = (self.Y2 - self.Y1)/(self.X2 - self.X1)
		else:
			self.m = null

		return self.m

	def sectionIntercept(self):
		self.b = self.Y2 - self.m*self.X2

		return self.b

	def distancePointToSection(self, P0):
		X0 = P0[0]
		Y0 = P0[1]

		num = abs((self.Y2 - self.Y1)*X0 - (self.X2 - self.X1)*Y0 + self.X2*self.Y1 - self.Y2*self.X1)
		den = math.sqrt((self.Y2 - self.Y1)**2 + (self.X2 - self.X1)**2)
		self.distance = num/den

		return self.distance

####################################################################################################
# Method Definitions
####################################################################################################
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
	"""
	This function draws `lines` with `color` and `thickness`.    
	Lines are drawn on the image inplace (mutates the image).
	"""
	leftLane = []
	rightLane = []
	leftLaneSlope = []
	rightLaneSlope = []
	leftLaneIntercept = []
	rightLaneIntercept = []
	startingY = img.shape[0]

	for eachLine in lines:
		for X1, Y1, X2, Y2 in eachLine:
			P1 = [X1, Y1]
			P2 = [X2, Y2]
			line = SectionApproximation(P1, P2)
			startingY = min(startingY, Y1, Y2)

			if (line.sectionSlope() > 0):
				rightLane.append(eachLine)
				rightLaneSlope.append(line.m)
				rightLaneIntercept.append(line.sectionIntercept())
			else:
				leftLane.append(eachLine)
				leftLaneSlope.append(line.m)
				leftLaneIntercept.append(line.sectionIntercept())

	startingXLeft = (startingY - np.mean(leftLaneIntercept))/np.mean(leftLaneSlope)
	endingXLeft = (img.shape[0] - np.mean(leftLaneIntercept))/np.mean(leftLaneSlope)
	startingXRight = (startingY - np.mean(rightLaneIntercept))/np.mean(rightLaneSlope)
	endingXRight = (img.shape[0] - np.mean(rightLaneIntercept))/np.mean(rightLaneSlope)
	endingY = img.shape[0]
	cv2.line(img, (int(startingXLeft), int(startingY)), (int(endingXLeft), int(endingY)), color, thickness)
	cv2.line(img, (int(startingXRight), int(startingY)), (int(endingXRight), int(endingY)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def removeBackground(image):
	"""
	This method isolates a road in an image by removing the background 
	"""
	ySize = image.shape[0]
	xSize = image.shape[1]
	apex = [xSize/2, ySize*3/8]
	leftBottom = [0, ySize]
	rightBottom = [xSize, ySize]
	region = np.array([[apex, leftBottom, rightBottom]], dtype=np.int32)
	filteredImg = region_of_interest(image, region)

	return filteredImg

def filterColor(image):
	"""
	This method isolates lane lines by blacking out regions below a RGB 
	threshold
	"""
	redThreshold = 200
	greenThreshold = 100
	blueThreshold = 0
	threshold = (image[:,:,0] < redThreshold) | \
	            (image[:,:,1] < greenThreshold) | \
	            (image[:,:,2] < blueThreshold)
	image[threshold] = [0,0,0]

	return image

def filterGray(image, threshold):
	"""
	This method isolates lane lines in a gray image by blacking out region
	below a threshold
	"""
	matchingField = image[:,:] < threshold
	image[matchingField] = 0

	return image

def process_image(image):
    blurImage = gaussian_blur(image, 5)
    removeBackgroundImage = removeBackground(blurImage)
    RGBFilterImage = filterColor(removeBackgroundImage)
    grayImage = grayscale(RGBFilterImage)
    grayFilterImage = filterGray(grayImage, 190)
    cannyImage = canny(grayFilterImage, 80, 150)
    houghImage = hough_lines(cannyImage, 1, np.pi/180, 5, 15, 3)
    weightedImage = weighted_img(houghImage, image)
    
    return weightedImage

####################################################################################################
# Main Program
####################################################################################################
'''
figure = plt.figure()

image = mpimg.imread('test_images/solidWhiteCurve.jpg')
figure.add_subplot(331)
plt.imshow(image)

blurImage = gaussian_blur(image, 5)
figure.add_subplot(332)
plt.imshow(blurImage)

removeBackgroundImage = removeBackground(blurImage)
figure.add_subplot(333)
plt.imshow(removeBackgroundImage)

RGBFilterImage = filterColor(removeBackgroundImage)
figure.add_subplot(334)
plt.imshow(RGBFilterImage)

grayImage = grayscale(RGBFilterImage)
figure.add_subplot(335)
plt.imshow(grayImage, cmap = 'gray')

grayFilterImage = filterGray(grayImage, 190)
figure.add_subplot(336)
plt.imshow(grayFilterImage, cmap = 'gray')

cannyImage = canny(grayFilterImage, 80, 150)
figure.add_subplot(337)
plt.imshow(cannyImage, cmap = 'gray')

houghImage = hough_lines(cannyImage, 1, np.pi/180, 5, 15, 3)
figure.add_subplot(338)
plt.imshow(houghImage)

weightedImage = weighted_img(houghImage, image)
figure.add_subplot(339)
plt.imshow(weightedImage)

plt.show()
'''

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)