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
#
from attrdict import AttrDict as Attributes

# internal
from vision.library  import VisionTool

from pprint import pprint
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
        self.__scale   = 1.3
        self.__angle   = 30
        self.__steps   = 10
        self.__history = []
    # -------------------------------------------------------------------------
    # property - pattern
    # -------------------------------------------------------------------------
    def pattern(self):
        return self.__pattern.astype(np.uint8)

    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def process_(self, frame):
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
    # process
    # -------------------------------------------------------------------------
    def process(self, frame):
        # -------------------------------------------------
        # angles
        # -------------------------------------------------
        def init_angles():
            return next_angles(-self.__angle, self.__angle)

        def next_angles(beg, end):
            stp = np.linspace(beg, end, self.__steps)
            stp = np.unique(stp.astype(int), axis=0)
            return stp
        
        # -------------------------------------------------
        # offset
        # -------------------------------------------------
        def init_offset():
            return np.array([0, 0])
        
        # -------------------------------------------------
        # offset
        # -------------------------------------------------
        def init_length(img, ref):
            rsh = np.array(ref.shape[:2])
            ish = np.array(img.shape[:2])
            beg = np.flip(ish * np.max(rsh/ish))
            return np.multiply(beg, self.__scale)

        # -------------------------------------------------
        # scales
        # -------------------------------------------------
        def init_scales(img, ref):
            rsh = np.array(ref.shape[:2])
            ish = np.array(img.shape[:2])
            beg = np.flip(ish * np.max(rsh/ish))
            return next_scales(beg, beg * self.__scale)

        def next_scales(beg, end):
            stp = np.linspace(beg, end, self.__steps)
            stp = np.unique(stp.astype(int), axis=0)
            return stp
        # --------------------------------------------------
        # process
        # --------------------------------------------------
        attr = Attributes(
            offset=init_offset(),
            length=init_length(frame, self.__pattern),
            angles=init_angles(),
            scales=init_scales(frame, self.__pattern))
        while True:
            # run 
            data = self.__iterate(frame, attr)
            # iterate
            pprint(data[0:10])
            break
        def print_region(data):
            sy, sx, = self.__pattern.shape[0:2]
            return [
                ((rx / fx, ry / fy), ((rx + sx) / fx, (ry + sy) / fy))
                for _, ry, rx, fy, fx in data
            ]
        return print_region(da)   
    # -------------------------------------------------------------------------
    # iterate
    # -------------------------------------------------------------------------
    def __iterate(self, data, args):
        # prepare input
        data = self.__rotate(data, args)
        # search pattern
        data = self.__resize(data, args)
        # search pattern
        data = self.__search(data, args)
        # return
        return data 
    
    # -------------------------------------------------------------------------
    # rotate
    # -------------------------------------------------------------------------
    def __rotate(self, frame, args):
        frames = []
        for angle in args.angles:
            frames.append((angle,
                iu.rotate(frame, angle=angle)))
        return frames
    
    # -------------------------------------------------------------------------
    # resize
    # -------------------------------------------------------------------------
    def __resize(self, frames, args):
        data = []
        for angle, frame in frames:
            for scale in args.scales:
                data.append((angle, scale,
                    self.__features(iu.resize(frame, *scale))))
        return data

    # -------------------------------------------------------------------------
    # search
    # -------------------------------------------------------------------------
    def __search(self, data, args):
        def find_region(data):
            print(data)
            sy, sx, = self.__pattern.shape[0:2]
            return [
                ((rx / fx, ry / fy), ((rx + sx) / fx, (ry + sy) / fy))
                for _, ry, rx, fy, fx in data
            ]
        # process
        out = []
        for angle, scale, frame in data:
            out += [ 
                (angle, scale, pos, cor)  
                for cor, pos in self.__convolve(frame)
            ]
        # sort and return
        out.sort(key = lambda x: x[3])
        return out
    
    def __convolve(self, img):
        return self.__convolve_base(
            img, 
            self.__pattern, 
            np.array([self.__steps, self.__steps]))


    # -------------------------------------------------------------------------
    # steps 1 - input preparation
    # -------------------------------------------------------------------------        
    def __prepare(self, data):
        return self.__prepare_input(data, self.__pattern, self.__scale)
    
    # -------------------------------------------------------------------------
    # steps 2 - search
    # -------------------------------------------------------------------------        
    def __search_(self, data):
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
    # tool - prepare pattern
    # -------------------------------------------------------------------------
    @staticmethod
    def __features(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        img.astype(float)
        return img
    
    # -------------------------------------------------------------------------
    # tool - convolution
    # -------------------------------------------------------------------------    
    @staticmethod
    def __convolve_base(img, ref, size):
        # compute parameters
        rsz = np.array(ref.shape[:2])
        isz = np.array(img.shape[:2])
        dif = (isz - rsz)
        stp = (dif / size + 1).astype(np.uint32)
        dsz = (dif / stp  + 1).astype(np.uint32)
        # output data
        out = []
        for i in range(dsz[0]):
            y = i * stp[0]
            for j in range(dsz[1]):
                x = j * stp[1]
                roi = img[y : y + rsz[0], x : x + rsz[1]]
                crr = np.corrcoef(roi.flatten(), ref.flatten())
                out.append((crr[0, 1], (y, x)))
        return out



    # -------------------------------------------------------------------------
    # tool - prepare pattern
    # -------------------------------------------------------------------------
    def __prepare_pattern(self, data, area=30000):
        # current shape
        shape = np.array(data.shape[:2])
        # updated shape
        shape = (shape * np.sqrt(area / np.product(shape))).astype(int)
        # reshaped pattern
        data  = iu.resize(data, shape[0], shape[1])
        # extract features
        data  = self.__features(data)
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
