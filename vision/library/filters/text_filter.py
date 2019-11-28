# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   text_filter.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################

# internal
from vision.library       import VisionFilter
from vision.library.tools import TextDetectionTool
from vision.library.tools import TextRecognitionTool

# external
from imutils.perspective  import four_point_transform
from numpy                import array, flip

# ################################################################################################
# ------------------------------------------------------------------------------------------------
# VisionFilterText
# ------------------------------------------------------------------------------------------------
# ################################################################################################
class TextFilter(VisionFilter):

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__(config)
        # load detection
        self.__detection   = TextDetectionTool()
        # load recognition
        self.__recognition = TextRecognitionTool()
    
    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    def process(self, frame):
        # crop by filter roi
        data = self.crop(frame)
        # localize texts 
        results = []
        for contour, roi in self.__localization(data):
            # translate roi
            label, level = self.__translate(roi)
            # save result
            results.append(VisionFilter.Result(
                label, level, VisionFilter.Region(contour[0], contour[1])))
        # select results
        return self.select(results)

    # -------------------------------------------------------------------------
    # step 1 - text localization
    # -------------------------------------------------------------------------         
    def __localization(self, data):
        output = []
        size = flip(array(data.shape[:2]))
        for roi in self.__detection.process(data):

            # padding = 2
            # roi[0][0] = max(0,       roi[0][0] - padding)
            # roi[0][1] = min(size[1], roi[0][1] + padding)
            
            # roi[1][0] = max(0, roi[1][0] - padding)
            # roi[1][1] = max(0, roi[1][1] - padding)
            
            # roi[2][0] = min(size[0], roi[2][0] + padding)
            # roi[2][1] = max(0      , roi[2][1] - padding)
            
            # roi[3][0] = min(size[0], roi[3][0] + padding)
            # roi[3][1] = min(size[1], roi[3][1] + padding)
           
            contour = array((roi.min(0), roi.max(0)-roi.min(0))) / size
            output.append((contour, four_point_transform(data, roi)))
        return output

    # -------------------------------------------------------------------------
    # step 2 - translate from img to text 
    # -------------------------------------------------------------------------         
    def __translate(self, data):
        return (self.__recognition.process(data), 1.0)
         
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################
