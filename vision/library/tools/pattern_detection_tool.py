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
        self.__scale   = [1.0, 1.3, 0.05]
        self.__history = []
    # -------------------------------------------------------------------------
    # property - pattern
    # -------------------------------------------------------------------------
    def pattern(self):
        return self.__pattern.astype(np.uint8)

    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def process(self, frame):
        def print_region(frame, region):
            from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
            sy, sx = frame.shape[:2]
            begin  = (int(sy * region[0][0]), int(sy * region[0][1]))
            end    = (int(sx * region[1][0]), int(sy * region[1][1]))
            rectangle(frame, begin, end, (255,0, 0), 2)
        # prepare input
        data = self.__prepare(frame)
        # search pattern
        for region in self.__search(data):
            print(region)
            print_region(frame, region)
        return frame 
    # -------------------------------------------------------------------------
    # steps 1 - input preparation
    # -------------------------------------------------------------------------        
    def __prepare(self, data):
        return self.__prepare_input(data, self.__pattern, self.__scale)
    
    # -------------------------------------------------------------------------
    # steps 2 - search
    # -------------------------------------------------------------------------        
    def __search(self, data):
        def find_region(data):
            print(data)
            sy, sx, = self.__pattern.shape[0:2]
            return [
                ((rx / fx, ry / fy), ((rx + sx) / fx, (ry + sy) / fy))
                for _, ry, rx, fy, fx in data
            ]
        # process
        out = np.concatenate([
            self.__convolve_base(each, self.__pattern, np.array([10, 10]))
            for each in data
        ])
        # sort and return
        return find_region(out[out[:,0].argsort()][-10:])
    # -------------------------------------------------------------------------
    # steps 3 - search
    # ------------------------------------------------------------------------- 
    def __select(self, data):
        pass
    # -------------------------------------------------------------------------
    # tool - prepare pattern
    # -------------------------------------------------------------------------
    @staticmethod
    def __extract_features(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)[:,:, 0:3]
        img.astype(float)
        return img

    # -------------------------------------------------------------------------
    # tool - prepare pattern
    # -------------------------------------------------------------------------
    def __prepare_pattern(self, data, area=30000):
        # current shape
        shape   = np.array(data.shape[:2])
        # updated shape
        reshape = (shape * np.sqrt(area / np.product(shape))).astype(int)
        # reshaped pattern
        data    = iu.resize(data, reshape[0], reshape[1])
        # extract features
        data    = self.__extract_features(data)
        # normalize and return
        return data

    # -------------------------------------------------------------------------
    # tool - prepare input
    # -------------------------------------------------------------------------
    def __prepare_input(self, img, ref, scale):
        rsz = np.array(ref.shape[:2])
        isz = np.array(img.shape[:2])
        # base shape
        ref = np.flip(isz * np.max(rsz/isz))
        # return a set of reshaped images
        return [
            self.__extract_features(iu.resize(img, *(ref * k).astype(int))) 
            for k in np.arange(*scale) 
        ]
    # -------------------------------------------------------------------------
    # tool - convolution
    # -------------------------------------------------------------------------
    # [correlation, y, x, d, r1, r2, r3]
    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def __convolve_base_(img, ref, size):
        # compute parameters
        rsz = np.array(ref.shape[:2])
        isz = np.array(img.shape[:2])
        dif = (isz - rsz)
        stp = (dif / size + 1).astype(np.uint32)
        dsz = (dif / stp  + 1).astype(np.uint32)
        # output data
        out = np.empty((dsz[0], dsz[1], 5))
        for i in nb.prange(dsz[0]):
            y = i * stp[0]
            for j in nb.prange(dsz[1]):
                x = j * stp[1]
                roi = img[y : y + rsz[0], x : x+ rsz[1]]
                cnv = roi * ref
                out[i, j] = np.append(np.array([cnv.sum() / roi.sum(), y, x]), isz)
        return out.reshape((-1, 5))

    @staticmethod
    #@nb.jit(nopython=True, parallel=True)
    def __convolve_base(img, ref, size):
        # compute parameters
        rsz = np.array(ref.shape[:2])
        isz = np.array(img.shape[:2])
        dif = (isz - rsz)
        stp = (dif / size + 1).astype(np.uint32)
        dsz = (dif / stp  + 1).astype(np.uint32)
        # output data
        out = np.empty((dsz[0], dsz[1], 5))
        for i in range(dsz[0]):
            y = i * stp[0]
            for j in range(dsz[1]):
                x = j * stp[1]
                roi = img[y : y + rsz[0], x : x + rsz[1]]
                crr = np.corrcoef(roi.flatten(), ref.flatten())
                out[i, j] = np.array([crr[0, 1], y, x, isz[0], isz[1]])
        return out.reshape((-1, 5))


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
    parser.add_argument('--images'  , '-i', type= str, default='./*.jpg')
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
    cv.namedWindow('test1'  , cv.WINDOW_NORMAL)
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
        #cv.imshow('pattern', tool.pattern())
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
