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
import cv2     as cv

# internal
from vision.library  import VisionTool

from pprint import pprint
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# Pattern Detection Tools
# ------------------------------------------------------------------------------------------------
# ################################################################################################

# -----------------------------------------------------------------------------
# resize image 
# -----------------------------------------------------------------------------
def _resize(data, ctxt):
    def process(scale):
        size = tuple(np.flip(scale))
        return cv.resize(data, size, interpolation=cv.INTER_AREA)
    return map(process, ctxt.scales)

# -----------------------------------------------------------------------------
# rotate image 
# -----------------------------------------------------------------------------
def _rotate(data, ctxt):       
    def process(angle):
        shape = np.flip(np.array(data.shape[:2]))
        return cv.warpAffine(data,
            cv.getRotationMatrix2D(tuple(shape/2), angle, 1.0), tuple(shape))
    return np.array([process(angle) for angle in ctxt.angles])

# -----------------------------------------------------------------------------
# select
# -----------------------------------------------------------------------------
def _select(data, ctxt):
    return data

# -----------------------------------------------------------------------------
# search
# -----------------------------------------------------------------------------
def _search(data, ctxt):
    from vision.library.helpers.vectorize  import sliding
    # using parameters
    axs = (-3, -2)
    stp = ctxt.steps
    ref = ctxt.pattern
    # sliding window
    data = sliding(data, axis=axs, step=stp, shape=np.take(ref.shape,axs))
    # mean 
    #data = data - np.mean(data, axis=axs, keepdims=True)
    ref  = ref  - np.mean(ref,  axis=axs, keepdims=True)
    aux  = data * ref
    aux = data
    print("1", aux.shape)
    aux  = np.sum(aux, axis=axs)
    print("2", aux.shape)
    return data

# -----------------------------------------------------------------------------
#  
# -----------------------------------------------------------------------------
def _process(data, ctxt):
    # rotate img
    data = _rotate(data, ctxt)
    print("rotate", data.shape)
    # select region 
    data = _select(data, ctxt)
    print("select", data.shape)
    # search pattern
    data = _search(data, ctxt)
    print("search", data.shape)
    return data.shape

# #################################################################################################
# -------------------------------------------------------------------------------------------------
# Pattern Detection Interface
# -------------------------------------------------------------------------------------------------
# #################################################################################################
class PatternDetectionTool(VisionTool):    
    class Context:
        def __init__(self, pattern):
            self.offset = np.array([0.0, 0.0])
            self.shape  = np.array([1.0, 1.0])
            self.angles = np.array([0, 0])
            self.scales = np.array([0, 0])
            self.steps  = np.array([1, 1])
            self.pattern= pattern
    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self, pattern):
        super().__init__()
        # settings
        self.__scale = 1.3
        self.__angle = 30
        self.__steps = 10
        self.__area  = 30000
        # properties
        self.__pattern = self.__prepare_pattern(pattern)

    # -------------------------------------------------------------------------
    # property - pattern
    # -------------------------------------------------------------------------
    def pattern(self):
        return self.__pattern.astype(np.uint8)
    
    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def process(self, data):
        from multiprocessing import Pool, cpu_count
        from functools       import partial

        data = iu.resize(data, *(103, 291))
        # context initialization 
        ctxt = self.__init_context(data)
        # parallelizing resize images
        with Pool(cpu_count()) as pool:
            from time import time
            # run process layers
            
            while self.__check_context(ctxt):
                ref = time()
                
                result = pool.map(
                    partial(_process, ctxt=ctxt), _resize(data, ctxt))

                print("----------------------", time()-ref)
                # update context
                ctxt = self.__update_context(ctxt)
                #break
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
        
        return print_region(data, find_region(data[-10:]))  

    # -------------------------------------------------------------------------
    # context initialization 
    # -------------------------------------------------------------------------
    def __init_context(self, img):
        # create context
        ctxt = self.Context(self.__pattern)
        # init scales
        rsh = np.array(self.__pattern.shape[:2])
        ish = np.array(img.shape[:2])
        begin  = ish * np.max(rsh / ish)
        scales = np.linspace(begin, begin * self.__scale, self.__steps)
        scales = np.unique(scales.astype(int), axis=0)
        ctxt.scales = scales
        # init angles
        angles = np.linspace(-self.__angle, self.__angle, self.__steps)
        angles = np.unique(angles.astype(int), axis=0)
        ctxt.angles = angles
        # init steps
        ctxt.steps = np.array([self.__steps, self.__steps])
        # return context
        return ctxt

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
    # resize
    # -------------------------------------------------------------------------
    def __resize(self, data, ctxt):
        return [
            self.__features(iu.resize(data, *np.flip(scale))) 
            for scale in ctxt.scales 
        ]

    # -------------------------------------------------------------------------
    # select
    # -------------------------------------------------------------------------
    def __select(self, data, ctxt):
        return data

    # -------------------------------------------------------------------------
    # helper - prepare image
    # -------------------------------------------------------------------------
    def __prepare_image(self, data):
        # rsh   = np.array(self.__pattern.shape[:2])
        # ish   = np.array(data.shape[:2])
        # shape = self.__scale * ish * np.max(rsh / ish) 
        # # reshaped pattern
        # data  = iu.resize(data, *np.flip(shape.astype(int)))
        return data
    
    # -------------------------------------------------------------------------
    # helper - prepare pattern
    # -------------------------------------------------------------------------
    def __prepare_pattern(self, data):
        # current shape
        shape = np.array(data.shape[:2])
        # updated shape
        shape = (shape * np.sqrt(self.__area / np.product(shape)))
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
