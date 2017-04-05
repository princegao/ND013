####################################################################################################
####################################################################################################
# Name: Lane Finding
# Coder: Janson Fong
# Description:
#	Given a video with lane lines, this script outputs a video with lane lines highlighted in red.
# Change the input and output name in the main program to execute script on a different video in 
# your directory. 
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

class Line:
	'''
	This class defines a line given two points. The slope and intercept of 
	a line can be calculated via the slope and intercept method
	'''
	def __init__(self, P1, P2):
		self.X1 = P1[0]
		self.Y1 = P1[1]
		self.X2 = P2[0]
		self.Y2 = P2[1]

	def slope(self):
		'''
		THis method returns the slope of a line defined by P1 and P2
		'''
		if (self.X2 - self.X1 != 0):
			self.m = (self.Y2 - self.Y1)/(self.X2 - self.X1)
		else:
			self.m = null

		return self.m

	def intercept(self):
		'''
		This method returns the intercept of a line defined by P1 and P2
		'''
		self.b = self.Y2 - self.m*self.X2

		return self.b

class FindLane:
	'''
	This class determines left and right lane given a set of hough lines
	sortByPoint and sortBySlope are two methods given to determine the 
	start and end points of a lane. A lane is represented as an instance 
	of the 'Line' class
	'''
	def __init__(self, lines):
		self.lines = lines 

	def sortByPoint(self):
		'''
		The sortByPoint method return the start and end points of lanes by 
		returning points with minimum and maximum y values respectively
		'''
		rightStartPoint = [0, 900]
		rightEndPoint = [0, 0]
		leftStartPoint = [0, 900]
		leftEndPoint = [0, 0]

		for eachLine in self.lines:
			for X1, Y1, X2, Y2 in eachLine:
				P1 = [X1, Y1]
				P2 = [X2, Y2]
				line = Line(P1, P2)
				slope  = line.slope()

				if (slope > 0):
					rightStartPoint = self.__findStartPoint(P1, P2, rightStartPoint)
					rightEndPoint = self.__findEndPoint(P1, P2, rightEndPoint)
				elif (slope < 0 and slope < -0.1):
					leftStartPoint = self.__findStartPoint(P1, P2, leftStartPoint)
					leftEndPoint = self.__findEndPoint(P1, P2, leftEndPoint)

		self.rightLane = Line(rightStartPoint, rightEndPoint)
		self.leftLane = Line(leftStartPoint, leftEndPoint)

	def sortBySlope(self):
		'''
		The sortBySlope method return the start and end points of lanes by
		averaging the slope and intercept of a set of lines. Using the averaged 
		slope and intercept, the start and end points of lanes are calculated 
		'''
		START_Y = 375
		END_Y = 900
		ZERO = 0.0
		rightLane = {}
		leftLane = {}

		for eachLine in self.lines:
			for X1, Y1, X2, Y2 in eachLine:
				P1 = [X1, Y1]
				P2 = [X2, Y2]
				line = Line(P1, P2)
				slope = line.slope()
				intercept = line.intercept()

				if (slope > 0):
					rightLane[slope] = intercept
				elif (slope < 0):
					leftLane[slope] = intercept

		# Ensuring lanes have enough points for filtering
		if (len(rightLane) > 1 and len(leftLane) > 1):
			# Removing extreme slopes
			rightLane.pop(max(rightLane))
			rightLane.pop(min(rightLane))
			leftLane.pop(max(leftLane))
			leftLane.pop(min(leftLane))

		# Checking for division by zero errors 
		if (len(rightLane) > 0 and len(leftLane) > 0):
			# Averaging slope
			rightSlope = self.__average(rightLane.keys())
			rightIntercept = self.__average(rightLane.values())
			leftSlope = self.__average(leftLane.keys())
			leftIntercept = self.__average(leftLane.values())

			# Calculating lane start, end point
			rightStartPoint = self.__calculateLaneCoor(rightSlope, rightIntercept, y = START_Y)
			rightEndPoint = self.__calculateLaneCoor(rightSlope, rightIntercept, y = END_Y)
			leftStartPoint = self.__calculateLaneCoor(leftSlope, leftIntercept, y = START_Y)
			leftEndPoint = self.__calculateLaneCoor(leftSlope, leftIntercept, y = END_Y)
		else:
			rightStartPoint = [1,1]
			rightEndPoint = [1,1]
			leftStartPoint = [1,1]
			leftEndPoint = [1,1]

		self.rightLane = Line(rightStartPoint, rightEndPoint)
		self.leftLane = Line(leftStartPoint, leftEndPoint)

	def __findStartPoint(self, P1, P2, startPoint):
		'''
		This private method returns the point with min y value
		'''
		if (P1[1] < P2[1]):
			if (P1[1] < startPoint[1]):
				return P1
			else:
				return startPoint
		else:
			if (P2[1] < startPoint[1]):
				return P2
			else:
				return startPoint

	def __findEndPoint(self, P1, P2, endPoint):
		'''
		This private method returns the point with max y value
		'''
		if (P1[1] > P2[1]):
			if (P1[1] > endPoint[1]):
				return P1
			else:
				return endPoint
		else:
			if (P2[1] > endPoint[1]):
				return P2
			else:
				return endPoint

	def __average(self, arrayList):
		'''
		This private method returns the average of an array
		'''
		return sum(arrayList)/len(arrayList)

	def __calculateLaneCoor(self, slope, intercept, x = None, y = None):
		'''
		This private method calculates the x or y value of a point given
		slope, intercept, and x or y
		'''
		if (x == None and y != None):
			x = int((y - intercept)/slope)
			return [x, y]
		elif (x != None and y == None):
			y = int(slope*x + intercept)
			return [x, y]

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
	lane = FindLane(lines)
	lane.sortBySlope()
	rightLane = lane.rightLane
	leftLane = lane.leftLane
	cv2.line(img, (rightLane.X1, rightLane.Y1), (rightLane.X2, rightLane.Y2), color, thickness)
	cv2.line(img, (leftLane.X1, leftLane.Y1), (leftLane.X2, leftLane.Y2), color, thickness)

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
	apex = [xSize/2, ySize/2]
	tolerance = 75
	leftBottom = [tolerance, ySize]
	rightBottom = [xSize - tolerance, ySize]
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
	'''
	This method returns the lanes of a image by first applying a gaussian blur,
	removing background details, filtering RGB colors, applying a grayscale,
	filtering a grayscale image, and applying canny transform. Red lane lines 
	are overlayed the original image.
	'''
	blurImage = gaussian_blur(image, 5)
	removeBackgroundImage = removeBackground(blurImage)
	RGBFilterImage = filterColor(removeBackgroundImage)
	grayImage = grayscale(RGBFilterImage)
	grayFilterImage = filterGray(grayImage, 190)
	cannyImage = canny(grayFilterImage, 80, 150)
	houghImage = hough_lines(cannyImage, 1, np.pi/180, 5, 15, 3)
	weightedImage = weighted_img(houghImage, image)

	return weightedImage

def laneFindVideo(title, inputVideo):
	'''
	This method returns a video with lane lines highlighted in red 
	'''
	video = title
	clip = VideoFileClip(inputVideo)
	videoClip = clip.fl_image(process_image)
	videoClip.write_videofile(video, audio=False)

	return video

####################################################################################################
# Main Program
####################################################################################################

laneFindVideo('white.mp4', 'solidWhiteRight.mp4')
laneFindVideo('yellow.mp4', 'solidYellowLeft.mp4')
laneFindVideo('extra.mp4', 'challenge.mp4')