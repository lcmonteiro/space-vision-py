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
            #return [0]
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
            beg = (ish * np.max(rsh/ish))
            return np.multiply(beg, self.__scale)

        # -------------------------------------------------
        # scales
        # -------------------------------------------------
        def init_scales(img, ref):
            rsh = np.array(ref.shape[:2])
            ish = np.array(img.shape[:2])
            beg = ish * np.max(rsh/ish)
            return next_scales(beg, beg * self.__scale)

        def next_scales(beg, end):
            stp = np.linspace(beg, end, self.__steps)
            stp = np.unique(stp.astype(int), axis=0)
            return stp
        # --------------------------------------------------
        # process
        # --------------------------------------------------
        attr = Attributes(
            rshape=np.array(self.__pattern.shape[:2]),
            offset=np.array([]),
            length=init_length(frame, self.__pattern),
            angles=init_angles(),
            scales=init_scales(frame, self.__pattern))
        print(attr)
        while True:
            # run 
            data = self.__iterate(frame, attr)
            # iterate
            pprint(data[-10:])
            break

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
        
        def print_correlation(frame, region):
            import matplotlib.pyplot as plt
            ish = region[1]
            off = region[2]
            rsh = np.array(self.__pattern.shape[0:2])
            #
            f = self.__features(iu.resize(frame, *np.flip(ish))) 
            #
            p = np.array([off, (rsh + off)], dtype=int) 
            print(rsh)
            print(p)
            #
            crop = f[p[0][0]:p[1][0], p[0][1]:p[1][1]]
            cv.imshow('test'  , crop)
            crop = np.reshape(crop, (-1, 3)) + 300
            print(crop.size)
            print(crop.shape)
            pattern = np.reshape(self.__pattern, (-1, 3))
            plt.plot(crop)
            plt.plot(pattern)
            plt.show()

        #print_correlation(frame, data[-1])  
        
        return print_region(frame, find_region(data[-10:]))  
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
                    self.__features(iu.resize(frame, *np.flip(scale)))))
        return data

    # -------------------------------------------------------------------------
    # search
    # -------------------------------------------------------------------------
    def __search(self, data, args):
        # process
        out = []
        for angle, scale, frame in data:
            out += self.__do_correlation(angle, scale, frame)
        # sort and return
        out.sort(key = lambda x: x[3])
        return out
    
    def __do_correlation(self, angle, scale, frame):
        size = np.array([self.__steps, self.__steps])
        return [ 
            (angle, scale, pos, cor)  
            for cor, pos in self.__correlation_v(frame, self.__pattern, size)
        ]
    # -------------------------------------------------------------------------
    # steps 1 - input preparation
    # -------------------------------------------------------------------------        
    def __prepare(self, data):
        return self.__prepare_input(data, self.__pattern, self.__scale)

    # -------------------------------------------------------------------------
    # tool - prepare pattern
    # -------------------------------------------------------------------------
    @staticmethod
    def __features(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        #img = img[:,:, 2:3]
        img.astype(float)
        return img
    
    # -------------------------------------------------------------------------
    # tool - correlation
    # -------------------------------------------------------------------------    
    @staticmethod
    def __correlation(img, ref, size):
        # compute parameters
        rsz = np.array(ref.shape[:2])
        isz = np.array(img.shape[:2])
        dif = (isz - rsz)
        stp = (dif / size + 1).astype(np.uint32)
        # output data
        out = []
        for y in range(0, dif[0], stp[0]):
            for x in range(0, dif[1], stp[1]):
                roi = img[y : y + rsz[0], x : x + rsz[1]]
                #crr = np.corrcoef(roi.flatten(), ref.flatten())[0,1]
                crr = np.sqrt(np.sum(np.power(
                    np.array([
                        np.corrcoef(roi[:,:,0].flatten(), ref[:,:,0].flatten())[0,1],
                        np.corrcoef(roi[:,:,1].flatten(), ref[:,:,1].flatten())[0,1],
                        np.corrcoef(roi[:,:,2].flatten(), ref[:,:,2].flatten())[0,1],
                        ((255 - np.abs(roi[:,:,0].flatten().mean() - ref[:,:,0].flatten().mean())) / 255) 
                    ]), 2
                )))
                # 
                out.append((crr, np.array([y, x])))
        return out

    # -------------------------------------------------------------------------
    # tool - correlation - vectorized 
    # img shape = (H, W, C)
    # ------------------------------------------------------------------------- 
    @staticmethod
    def __correlation_v(img, ref, step):
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




        

        
        
        return img



    # -------------------------------------------------------------------------
    # tool - prepare pattern
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
# python -m vision.library.tools.pattern_detection_tool_2 -t /c/Workspace/gen5/template.jpg -i "c:\Workspace\gen5\learn\*"