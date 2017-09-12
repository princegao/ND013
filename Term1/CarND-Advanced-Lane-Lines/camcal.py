###############################################################################
###############################################################################
# Name: camcal.py
# Coder: Janson Fang
# Description:
#   This modules calibrates a camera using images of chesboard taken at
# different angles
###############################################################################
###############################################################################
# Libraries and Modules
###############################################################################
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from log import logger
###############################################################################
# Constants
###############################################################################
###############################################################################
# Class Definitions
###############################################################################
class CamCal:
    '''Calibrate camera given images of chessboard taken at different angles

    Attributes:
        images (list): A list of chessboard image file paths
    '''
    def __init__(self, images):
        self.images = images
        self.objPts = []
        self.imgPts = []

    def initObjPts(self, x = 9, y = 6):
        '''Initialize chessboard object points i.e (0,0,0), (1,0,0),...,(6,5,0)

        Since images of chessboard are on a flat surface, the z dimension of
        all object points are zero.

        Args:
            x (int): Horizontal shape of chessboard
            y (int): Vertical shape of chessboard

        Returns:
            objPts (np.ndarray): An array of object points for a chessboard
            with xy corners
        '''
        objPts = np.zeros((x*y,3), np.float32)
        objPts[:,:2] = np.mgrid[0:x,0:y].T.reshape(-1,2)

        return objPts

    def findCorners(self, x = 9, y = 6):
        '''Find chessboard corners for all images in self.images

        Args:
            x (int): Horizontal shape of chessboard
            y (int): Vertical shape of chessboard

        Returns:
            If chessboard corners are found, object points and image points
            are appended in self.objPts and self.imgPts respectively
        '''
        objPts = self.initObjPts()
        for fileName in self.images:
            logger.info('Reading {} image file'.format(fileName))
            img = cv2.imread(fileName)
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            logger.info('Finding corners in {} image'.format(fileName))
            ret, corners = cv2.findChessboardCorners(grayImg, (x,y), None)

            if(ret):
                logger.info('Corners found for {} image.'.format(fileName) +
                            'Appending object points and image points')
                self.objPts.append(objPts)
                self.imgPts.append(corners)

                img = cv2.drawChessboardCorners(img, (x,y), corners, ret)
                cv2.imshow(fileName, img)

        cv2.destroyAllWindows()
###############################################################################
# Method Definitions
###############################################################################
def main():
    '''Calibrate camera using chessboard images

    Returns:
        Return an instance of CamCal
    '''
    logger.info('Starting camera calibration sequence')
    images = glob.glob(r'images/calibration/calibration*.jpg')
    calibrate = CamCal(images)
    calibrate.findCorners()

    return calibrate
###############################################################################
# Main Script
###############################################################################
if __name__ == "__main__":
    calibrate = main()