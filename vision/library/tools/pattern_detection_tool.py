# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:  template_detection_tool.py
# Author: Luis Monteiro
#
# Created on nov 17, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################

# external
import numba   as nb
import numpy   as np
import imutils as iu

# internal
from vision.library                import VisionTool

# ################################################################################################
# ------------------------------------------------------------------------------------------------
# TextDetection 
# ------------------------------------------------------------------------------------------------
# ################################################################################################
class PatternDetectionTool(VisionTool):
    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self, pattern):
        super().__init__()
        # properties
        self.__pattern = self.__prepare_pattern(pattern)

        self.__history = ((0,0,0), (0,0), (0,0)) 
    # -------------------------------------------------------------------------
    # property - pattern
    # -------------------------------------------------------------------------
    def pattern(self):
        return self.__pattern.astype(np.uint8)

    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def process(self, frame):
        # prepare input
        data = self.__prepare(frame)
        # search pattern
        return self.__search(data)
    
    # -------------------------------------------------------------------------
    # steps 1 - input preparation
    # -------------------------------------------------------------------------        
    def __prepare(self, data):
        return data.astype(float)
    
    # -------------------------------------------------------------------------
    # steps 2 - search
    # -------------------------------------------------------------------------        
    def __search(self, data):
        cv.imshow('test', data.astype(np.uint8))
        cv.imshow('test', iu.rotate(data, 10).astype(np.uint8))
        cv.imshow('test', iu.resize(data, 200, 100).astype(np.uint8))
        self.__convolve(data, self.__pattern)        
        return data
    # -------------------------------------------------------------------------
    # tool - prepere pattern
    # -------------------------------------------------------------------------
    @staticmethod
    def __prepare_pattern(data, area=30000):
        # current shape
        shape   = np.array(data.shape[:2])
        # updated shape
        reshape = (shape * np.sqrt(area / np.product(shape))).astype(int)
        # return a reshaped pattern
        return iu.resize(data, reshape[0], reshape[1]).astype(float)
    # -------------------------------------------------------------------------
    # tool - convolution
    # -------------------------------------------------------------------------
    @staticmethod
    @nb.jit(nopython=True, nogil=True, parallel=True)
    def __convolve(img, ref):
        rsz = np.array(ref.shape[:2])
        isz = np.array(img.shape[:2])
        dsz = isz - rsz + 1 
        out = np.empty((dsz[0], dsz[1]))
        for y in nb.prange(dsz[0]):
            for x in nb.prange(dsz[1]):
                roi = img[y:y+rsz[0], x:x+rsz[1]]
                cnv = roi * ref
                out[y, x] = cnv.sum() / roi.sum()
        return out

# ################################################################################################
# ------------------------------------------------------------------------------------------------
# Test
# ------------------------------------------------------------------------------------------------
# ################################################################################################
if __name__ == '__main__':
    from argparse import ArgumentParser
    from glob     import glob 
    import cv2    as cv 
    # -----------------------------------------------------------------------------------
    # parse parameters
    # -----------------------------------------------------------------------------------
    parser = ArgumentParser()
    # template
    parser.add_argument('--template', '-t', type= str, default='./template.jpg')
    # images
    parser.add_argument('--images'  , '-i', type= str, default = './*.jpg')
    # parse
    args = parser.parse_args()

    # -----------------------------------------------------------------------------------
    #  initialization
    # -----------------------------------------------------------------------------------
    # windows
    cv.namedWindow('image'  , cv.WINDOW_NORMAL)
    cv.namedWindow('pattern', cv.WINDOW_NORMAL)
    cv.namedWindow('match'  , cv.WINDOW_NORMAL)
    cv.namedWindow('test'  , cv.WINDOW_NORMAL)
    # pattern
    pattern = cv.imread(args.template)
    # tool
    tool = PatternDetectionTool(pattern=pattern)

    # -----------------------------------------------------------------------------------
    # process
    # -----------------------------------------------------------------------------------
    for img in glob(args.images):
        print(img)
        # read image
        image = cv.imread(img)
        # find pattern in image
        match = tool.process(image)
        # print
        cv.imshow('pattern', tool.pattern())
        cv.imshow('image'  , image  ) 
        cv.imshow('match'  , match  )
        # check
        cv.waitKey(0)
cv.destroyAllWindows()
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################
