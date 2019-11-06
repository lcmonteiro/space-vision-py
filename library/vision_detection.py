#
# ------------------------------------------------------------------------------------------------
# File:   vision_detection.py
# Author: Luis Monteiro
#
# Created on oct 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------
# VisionDetection 
# ------------------------------------------------------------------------------------------------
#
class VisionDetection:
    #
    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------
    #
    def __init__(self, config):
        #
        # -------------------------------------------------
        # variables
        # -------------------------------------------------
        # command containers  
        #
        __select = []
        #
        # filter selected  
        #
        __selected = {}
        #
        # filter data base 
        #
        __filters = []
    #
    # ------------------------------------------------------------------------
    # serve
    # ------------------------------------------------------------------------
    #
    def serve(self, callback):
        callback('id','ssss')
        pass
    # ------------------------------------------------------------------------
    # write commands
    # ------------------------------------------------------------------------
    #
    def set_filter(self, id, level=1.0, delta=0):
        #std::lock_guard<std::mutex> lock(__locker);
        #__select.emplace_back(id, level);
        return self
    def clr_filter(self, id):
        #std::lock_guard<std::mutex> lock(__locker);
        #__select.emplace_back(id, -1);
        return self
    def set_filters(self, level=1.0, delta=0):
        #std::lock_guard<std::mutex> lock(__locker);
        #for(auto& f : __filters)
        #    __select.emplace_back(std::get<0>(f), level);
        return self
    def clr_filters(self):
        #std::lock_guard<std::mutex> lock(__locker);
        #for(auto& f : __filters)
        #    __select.emplace_back(std::get<0>(f), -1);
        return self
    #
    # ------------------------------------------------------------------------
    # read commands
    # ------------------------------------------------------------------------
    #
    def _read_commands(self):
        pass
        #return std::move(__select);
    #
    # ------------------------------------------------------------------------
    # load filters
    # ------------------------------------------------------------------------
    #
    def _load_filters(self, input):
        pass
#
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
#
