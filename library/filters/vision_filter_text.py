# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   vision_filter_text.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################
#
# internal
from library             import VisionFilter
from library.models      import VisionModelTextEAST
# external
from imutils.perspective import four_point_transform
from numpy               import array, flip
#
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# VisionFilterText
# ------------------------------------------------------------------------------------------------
# ################################################################################################
class VisionFilterText(VisionFilter):
    #
    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    #
    def __init__(self, config):
        super().__init__(config)
        # load model for text detection
        self.__model = VisionModelTextEAST()
    #
    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    #
    def process(self, frame):
        # crop by filter roi
        data = self.crop(frame)
        # localize texts 
        results = []
        for contour, roi in self.__localization(data):
            # recognition roi
            label, level = self.__recognition(roi)
            # save result
            results.append(VisionFilter.Result(
                label, level, VisionFilter.Region(contour[0], contour[1])))
        # select results
        return self.select(results)
    #
    # -------------------------------------------------------------------------
    # step 1 - text localization
    # -------------------------------------------------------------------------
    #         
    def __localization(self, data):
        output = []
        size = flip(array(data.shape[:2]))
        for roi in self.__model.process(data):
            contour = array((roi.min(0), roi.max(0)-roi.min(0))) / size
            output.append((contour, four_point_transform(data, roi)))
        return output
    #
    # -------------------------------------------------------------------------
    # step 2 - text recognition
    # -------------------------------------------------------------------------
    #         
    def __recognition(self, data):
        return '%d'%(data[0][0][0]), '%d'%(data[0][0][1]) 
        
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################
