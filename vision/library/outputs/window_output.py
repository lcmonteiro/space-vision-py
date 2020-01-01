# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   window_output.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################
# external
from cv2            import imshow                  
from cv2            import destroyWindow                  
from cv2            import namedWindow, WINDOW_NORMAL                   
from cv2            import rectangle, putText, FONT_HERSHEY_SIMPLEX
from numpy          import array, flip
from seaborn        import color_palette
# internal
from vision.library import VisionOutput
# #############################################################################
# -----------------------------------------------------------------------------
# VisionOutput 
# -----------------------------------------------------------------------------
# #############################################################################
class WindowOutput(VisionOutput):

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self, id='vision output'):
        super().__init__(id)
        # create output window
        namedWindow(self._id, WINDOW_NORMAL)

    # -------------------------------------------------------------------------
    # destroy
    # -------------------------------------------------------------------------
    def __del__(self):
        try:
            destroyWindow(self._id)
        except:
            pass
    
    # -------------------------------------------------------------------------
    # override flush
    # -------------------------------------------------------------------------
    def flush(self, observer):
        # colors generation
        colors = array(color_palette(None, len(self._filter))) * 255
        # print regions
        for (id, (region, results)), color in zip(self._filter.items(), colors):
            self.__print_region(id, region, flip(color))        
            # print detections    
            for result in results:
                self.__print_detection(region, result, flip(color))
        imshow(self._id, self._frame)
         # process base flush
        return super().flush(observer)
    
    # -------------------------------------------------------------------------
    # print tools
    # -------------------------------------------------------------------------
    def __print_region(self, name, region, color):
        f_size = flip(array(self._frame.shape[:2]))
        begin  = tuple(region.begin(f_size))
        end    = tuple(region.end  (f_size))
        rectangle(self._frame, begin, end, color, 2)
        putText(self._frame, name, begin, FONT_HERSHEY_SIMPLEX, 0.8, color, 10)
        putText(self._frame, name, begin, FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    def __print_detection(self, region, result, color):
        f_size = flip(array(self._frame.shape[:2]))
        r_size = region.size(f_size)
        begin  = tuple(result.region().begin(r_size) + region.begin(f_size))
        end    = tuple(result.region().end  (r_size) + region.begin(f_size))
        rectangle(self._frame, begin, end, color, 1)
        putText(self._frame, result.label(), begin, FONT_HERSHEY_SIMPLEX, 0.5, color, 5)
        putText(self._frame, result.label(), begin, FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# #################################################################################################
# -------------------------------------------------------------------------------------------------
# End
# -------------------------------------------------------------------------------------------------
# #################################################################################################