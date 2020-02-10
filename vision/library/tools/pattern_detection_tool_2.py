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
    def process(self, img):
        # resize & feature extration
        data = self.__prepare_image(img)
        # context initialization 
        ctxt = self.__init_context(data)
        # run process layers
        while self.__check_context(ctxt):
            # rotate img
            data = self.__rotate(data, ctxt)
            # resize img
            data = self.__resize(data, ctxt)
            # select region 
            data = self.__select(data, ctxt)
            # search pattern
            data = self.__search(data, ctxt)
            # update context
            ctxt = self.__update_context(ctxt)
            break
        # ------------------------------------------------------
        #
        # -----------------------------------------------------
        def find_region(data):
            rsh = np.array(self.__pattern.shape[0:2])
            out = []
            for _, ish, off, _ in data:
                beg = (off / ish)
                end = (off + rsh) / ish
                out.append(np.array([beg, end]))
            return out

        def print_region(frame, region):
            from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
            region = region * np.array(frame.shape[:2])
            for r in region.astype(int):
                p1 = tuple(np.flip(r[0]))
                p2 = tuple(np.flip(r[1]))
                rectangle(frame, p1, p2, (255,0, 0), 2)
        
        return print_region(frame, find_region(data[-10:]))  

    # -------------------------------------------------------------------------
    # context initialization 
    # -------------------------------------------------------------------------
    def __init_context(self, img):
        # init angles
        angles = np.linspace(-self.__angle, self.__angle, self.__steps)
        angles = np.unique(angles.astype(int), axis=0)
        # init scales
        rsh = np.array(ref.shape[:2])
        ish = np.array(img.shape[:2])
        beg = ish * np.max(rsh / ish)
        scales = np.linspace(beg, beg * self.__scale, self.__steps)
        scales = np.unique(stp.astype(int), axis=0)
        # init offset 
        offset = np.array([1.0, 1.0])
        # init shape
        shape  = np.array([1.0, 1.0])
        # create context
        return Attributes(
            offset=offset, shape=shape, angles=angles, scales=scales)

    # -------------------------------------------------------------------------
    # context verification 
    # -------------------------------------------------------------------------
    def __check_context(self, ctxt):
        return True

    # -------------------------------------------------------------------------
    # context update 
    # -------------------------------------------------------------------------
    def __update_context(self, ctxt):
        return ctxt

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
                    self.__features(iu.resize(frame, *np.flip(scale)))))
        return data

    # -------------------------------------------------------------------------
    # search
    # -------------------------------------------------------------------------
    def __search(self, data, args):
        from vision.library.helpers.vectorize  import sliding_window
        # sliding window
        img = sliding_window(img, step, ref.shape)
        print(img.shape)
        # correlation
        ref = ref - np.mean(ref)
        print(ref.shape)
        
        mean = np.mean(img, axis=(3, 4), keepdims=True)
        print(mean.shape)
        print(mean)

        img = img - np.mean(img, axis=(3, 4), keepdims=True)
        print(img.shape)
        print(img)

        acc = np.sum(img**2, axis=(3,4))
        print(acc.shape)
        print(acc)
        print(np.sqrt(acc[0,:,:,0]))

        
        img = img[0,0,0,:,:, 0]
        img = img - np.mean(img)
        img = np.sum(img**2)
        print(np.sqrt(img))

        # process
        out = []
        return out
    

    # -------------------------------------------------------------------------
    # steps 1 - input preparation
    # -------------------------------------------------------------------------        
    def __prepare(self, data):
        return self.__prepare_input(data, self.__pattern, self.__scale)


    
    # -------------------------------------------------------------------------
    # helper - prepare pattern
    # -------------------------------------------------------------------------
    def __prepare_pattern(self, data, area=30000):
        # current shape
        shape = np.array(data.shape[:2])
        # updated shape
        shape = (shape * np.sqrt(area / np.product(shape)))
        # reshaped pattern
        data  = iu.resize(data, *np.flip(shape.astype(int)))
        # extract features
        data  = self.__features(data)
        # normalize and return
        return data
    
    # -------------------------------------------------------------------------
    # helper - extract features
    # -------------------------------------------------------------------------
    @staticmethod
    def __features(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        img.astype(float)
        return img

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
    #cv.namedWindow('match'  , cv.WINDOW_NORMAL)
    #cv.namedWindow('test'  , cv.WINDOW_NORMAL)
    #cv.namedWindow('test1'  , cv.WINDOW_NORMAL)
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
        cv.imshow('pattern', pattern)
        cv.imshow('image'  , image) 
        #cv.imshow('match'  , match  )
        # check
        cv.waitKey(0)
cv.destroyAllWindows()
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################
