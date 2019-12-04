# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   vision_output.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################
#
# -----------------------------------------------------------------------------
# VisionOutput 
# -----------------------------------------------------------------------------
#
class VisionOutput:
    # ---------------------------------------------------------------
    # initialization
    # ---------------------------------------------------------------
    def __init__(self, id):
        self._id        = id
        self._frame     = []
        self._filter    = {}
        self._detection = {}
    # ---------------------------------------------------------------
    # interfaces
    # ---------------------------------------------------------------
    def write_frame(self, frame):
        self._frame     = frame
        self._filter    = {}
    def write_filter(self, id, region, result):
        self._filter[id]=(region, result)
    def flush(self, observer):
        if observer:
            for id, filter in self._filter.items():
                observer(id, filter[1])
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################