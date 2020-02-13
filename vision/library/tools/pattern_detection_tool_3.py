# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:  template_detection_tool.py
# Author: Luis Monteiro
#
# Created on nov 17, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################

# external
import numba           as nb
import numpy           as np
import imutils         as iu
import cv2             as cv 
import multiprocessing as mp 

# internal
from vision.library  import VisionTool

from pprint import pprint
from time import time
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# Pattern Detection - Tools
# ------------------------------------------------------------------------------------------------
# ################################################################################################
# -----------------------------------------------------------------------------
# likelihood
# -----------------------------------------------------------------------------
def _likelihood(hdr, img, ref, num):
    # compute parameters
    rsz = np.array(ref.shape[:2])
    isz = np.array(img.shape[:2])
    dif = (isz - rsz + 1)
    stp = (dif / num + 1).astype(np.uint32)
    # output data
    out = []
    for y in range(0, dif[0], stp[0]):
        for x in range(0, dif[1], stp[1]):
            # crop
            win = img[y : y + rsz[0], x : x + rsz[1]]
            # flatten
            w_0, r_0 = win[:,:,0].flatten(), ref[:,:,0].flatten()
            w_1, r_1 = win[:,:,1].flatten(), ref[:,:,1].flatten()
            w_2, r_2 = win[:,:,2].flatten(), ref[:,:,2].flatten()
            # similarity
            crr = np.sqrt(np.sum(np.power(
                np.array([
                    np.corrcoef(w_0, r_0)[0,1],
                    np.corrcoef(w_1, r_1)[0,1],
                    np.corrcoef(w_2, r_2)[0,1],
                    ((255 - np.abs(w_0.mean() - r_0.mean())) / 255), 
                    ((255 - np.abs(w_1.mean() - r_1.mean())) / 255),
                    ((255 - np.abs(w_2.mean() - r_2.mean())) / 255) 
                ]), 2
            )))
            # update
            out.append(hdr + [np.array([y, x]), crr])
    return out

# -------------------------------------------------------------------------
# rotate
# -------------------------------------------------------------------------
def _rotate(data, ctxt):
    frames = []
    header, frame = data
    for angle in ctxt.angles:
        frames.append((header+[angle], iu.rotate(frame, angle=angle)))
    return frames

# -------------------------------------------------------------------------
# resize
# -------------------------------------------------------------------------
def _resize(data, ctxt):
    frames = []
    header, frame = data
    for scale in ctxt.scales:
        frames.append((header+[scale], iu.resize(frame, *np.flip(scale))))
    return frames

# -------------------------------------------------------------------------
# search
# -------------------------------------------------------------------------
def _search(data, ctxt):
    # process
    out = []
    for header, frame in data:
        out += _likelihood(header, frame, ctxt.pattern, ctxt.steps)
    # sort and return
    #print(out[0])
    #out.sort(key = lambda x: x[-1])
    return out

# -------------------------------------------------------------------------
# iterate
# -------------------------------------------------------------------------
def _process(data, ctxt):
    ref = time()
    key = data[0]
    # search pattern
    data = _resize(data, ctxt)
    # search pattern
    data = _search(data, ctxt)
    print(key, len(data)," =2--------------------= ", time()-ref)
    # return
    return data

# ################################################################################################
# ------------------------------------------------------------------------------------------------
# Pattern Detection - Interface
# ------------------------------------------------------------------------------------------------
# ################################################################################################
class PatternDetectionTool(VisionTool):
    # -------------------------------------------------------------------------
    # Descriptor 
    # -------------------------------------------------------------------------
    class Descriptor:
        def __init__(self):
            self.angles = [0, 0]
            self.bpoint = [0.0, 0.0]
            self.epoint = [1.0, 1.0]
            self.areas  = [1000, 10000]
    # -------------------------------------------------------------------------
    # Reference 
    # -------------------------------------------------------------------------
    class References:
        def __init__(self):
            self.bpoint = np.array([0.0, 0.0])
            self.epoint = np.array([1.0, 1.0])
            self.angles = np.array([0, 0])
            self.scales = np.array([0, 0])

    # -------------------------------------------------------------------------
    # Context 
    # -------------------------------------------------------------------------
    class Context:
        def __init__(self, pattern):
            self.angles = np.array([0, 0])
            self.scales = np.array([0, 0])
            self.steps  = np.array([1, 1])
            self.pattern= pattern

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self, pattern):
        super().__init__()
        # properties
        self.__model = pattern
        self.__steps   = 8
        # area
        self.__area_limit  = [1000, 10000] 
        self.__area_steps  = 4
        # scale
        self.__scale_limit = [1, 1.3]
        self.__scale_steps = 8
        # angle
        self.__angle_limit = [-30, 30]
        self.__angle_steps = 8
        # slide
        self.__slide_steps = 8
        # tools
        self.__pool = mp.Pool(processes=mp.cpu_count()) 

    # -------------------------------------------------------------------------
    # destroy 
    # -------------------------------------------------------------------------
    def __del__(self):
        self.__pool.close()

    # -------------------------------------------------------------------------
    # property - pattern
    # -------------------------------------------------------------------------
    def pattern(self):
        return self.__model.astype(np.uint8)

    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def process(self, frame):
        from functools import partial
        des = self.__create_descriptor()
        print("descriptor")
        pprint(des.__dict__)
        # check resolutions
        for area in des.areas:
            # prepare model 
            mod = self.__prepare_model(self.__model, area)
            # prepare input
            inp = self.__prepare_input(frame, mod, des)
            # build references
            ref = self.__create_references(inp, mod, des )
            print("references")
            pprint(ref.__dict__)
            # search process
            aux = self.References() 
            while self.__check_references(aux, ref):
                start = time()
                # reference backup
                aux = ref
                # rotate input
                dat = self.__rotate_data(inp, ref)
                # select region 
                dat = self.__select_data(inp, ref)
                # build context from references
                txt = self.__create_context(mod, ref)
                print("context")
                pprint(txt.__dict__)
                # process data
                res = self.__pool.map(partial(_process, ctxt=txt), dat)
                # select results
                res = self.__select_results(res, txt)
                # update references
                ref = self.__update_references(res, mod)
                
                print("references")
                pprint(ref.__dict__)

                print("****************", time()-start)
            # update descriptor 
            self.__update_descriptor(des, ref)
        # extract output
        return self.__select_output(frame, ref)

    # -------------------------------------------------------------------------
    # prepare model
    # -------------------------------------------------------------------------
    def __prepare_model(self, data, area=10000):
        # current shape
        shape = np.array(data.shape[:2])
        # updated shape
        shape = (shape * np.sqrt(area / np.product(shape)))
        # reshaped model
        frame = iu.resize(data, *np.flip(shape.astype(int)))
        # return features
        return self.__features(frame)

    # -------------------------------------------------------------------------
    # prepare input
    # -------------------------------------------------------------------------
    def __prepare_input(self, image, model, des):
        # model shape
        msh = np.array(model.shape[:2])
        ish = np.array(image.shape[:2])
        # updated shape
        ush = ish * np.max((msh * np.max(des.scales)) / ish)
        # reshaped input
        img = iu.resize(image, *np.flip(ush.astype(int)))
        # return features
        return self.__features(img)
    
    # -------------------------------------------------------------------------
    # prepare data
    # -------------------------------------------------------------------------
    def __prepare_data(self, data, ctxt):
        return _rotate(([], data), ctxt)

    # -------------------------------------------------------------------------
    # build decriptor 
    # -------------------------------------------------------------------------
    def __create_descriptor(self):
        des = self.Descriptor()
        # init angles
        des.angles = np.array(self.__angle_limit)
        # init scales
        des.scales = np.array(self.__scale_limit)
        # init area
        areas = np.linspace(*self.__area_limit, self.__area_steps)
        areas = np.unique(areas.astype(int), axis=0)
        des.areas = areas
        # return descriptor
        return des

    # -------------------------------------------------------------------------
    # update descriptor 
    # -------------------------------------------------------------------------
    def __update_descriptor(self, des, ref):
        # update angles
        des.angles = ref.angles
        # update scales
        des.scales = ref.scales


    # -------------------------------------------------------------------------
    # build references 
    # -------------------------------------------------------------------------
    def __create_references(self, image, model, des):
        # create references
        refs = self.References()
        # init angles
        refs.angles = des.angles
        # init scales
        msh = np.array(model.shape[:2])
        ish = np.array(image.shape[:2])
        print("msh=", msh)
        print("ish=", ish)
        print("mmm=", np.max(msh / ish))
        beg = ish * np.max(msh / ish)
        refs.scales = np.array([beg * scale for scale in des.scales])
        # return references
        return refs

    # -------------------------------------------------------------------------
    # references verification 
    # -------------------------------------------------------------------------
    def __check_references(self, ref0, ref1):
        return False if \
            np.array_equal(ref0.bpoint, ref1.bpoint) and \
            np.array_equal(ref0.epoint, ref1.epoint) and \
            np.array_equal(ref0.scales, ref1.scales) and \
            np.array_equal(ref0.scales, ref1.scales) else True

    # -------------------------------------------------------------------------
    # references update 
    # -------------------------------------------------------------------------
    def __update_references(self, result, model):
        # create references container
        refs = self.References()
        # update begin and end points
        aux = []
        msh = np.array(model.shape[0:2])
        for _, ish, off, _ in result:
            beg = (off / ish)
            end = (off + msh) / ish
            aux.append(np.array([beg, end]))
        aux = np.array(aux)
        refs.bpoint=np.min(aux, axis=(0, 1))
        refs.epoint=np.max(aux, axis=(0, 1))
        # update angles
        a_val = np.array(list(map(lambda x: x[0], result)))
        a_min = a_val[np.argmin(a_val)]
        a_max = a_val[np.argmax(a_val)]
        refs.angles = np.array([a_min, a_max])
        # update scales
        s_val = np.array(list(map(lambda x: x[1], result)))
        s_min = s_val[np.argmin(s_val)]
        s_max = s_val[np.argmax(s_val)]
        refs.scales = np.array([s_min, s_max])
        # return updated references
        return refs

    # -------------------------------------------------------------------------
    # build context
    # -------------------------------------------------------------------------
    def __create_context(self, model, refs):
        # create context
        ctxt = self.Context(model)
        # init angles
        angles = np.linspace(*refs.angles, self.__angle_steps)
        angles = np.unique(angles.astype(int), axis=0)
        ctxt.angles = angles
        # init scales
        scales = np.linspace(*refs.scales, self.__scale_steps)
        scales = np.unique(scales.astype(int), axis=0)
        ctxt.scales = scales
        # init search
        #for s in scales:

        ctxt.steps = np.array([self.__steps, self.__steps])
        # return context
        return ctxt

    # -------------------------------------------------------------------------
    # select results 
    # -------------------------------------------------------------------------
    def __select_results(self, res, ctxt):
        from functools import reduce
        from operator  import iconcat
        # reduce
        res = reduce(iconcat, res, [])
        # sort
        res.sort(key = lambda x: x[-1])
        # select  
        return res[-10:]

    # -------------------------------------------------------------------------
    # select output
    # -------------------------------------------------------------------------
    def __select_output(self, frame, refs):
        def print_region(frame, region):
            from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
            sh = np.array(frame.shape[:2])
            p1 = tuple(np.flip(sh * refs.bpoint).astype(int))
            p2 = tuple(np.flip(sh * refs.epoint).astype(int))
            rectangle(frame, p1, p2, (255,0, 0), 2)
        return print_region(frame, refs)  

    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------
    def debug(self, data, frame):
        def find_region(data):
                rsh = np.array(self.__model.shape[0:2])
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
    # -------------------------------------------------------------------------
    # features
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
    from os       import _exit
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
    exit(0)
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################
# python -m vision.library.tools.pattern_detection_tool_2 -t /c/Workspace/gen5/template.jpg -i "c:\Workspace\gen5\learn\*"