# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   vision_detection.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################

# external imports
from cv2      import waitKey    as wait
from logging  import getLogger  as logger

# internal imports
from .vision_config import VisionConfigFilter

# ################################################################################################
# ------------------------------------------------------------------------------------------------
# VisionDetection 
# ------------------------------------------------------------------------------------------------
# ################################################################################################
class VisionDetector:

    # -----------------------------------------------------------------------------------
    # initialization
    # -----------------------------------------------------------------------------------
    def __init__(self, config, input, output):
        # -------------------------------------------------
        # variables
        # -------------------------------------------------
        # command containers  
        self.__select = []
        # load input 
        self.__input  = input
        # load output
        self.__output = output
        # load logger
        self.__logger = logger()
        # filter data base 
        self.__filters = self._load_filters(config)
    
    # -----------------------------------------------------------------------------------
    # serve
    # -----------------------------------------------------------------------------------
    def serve(self, observer):

        # check vision input
        if not self.__input.good():
            raise RuntimeError("vision input open fail ...")

        # image processing
        selected = {}
        while wait(1000) < 0:
            # read input frame
            frame = self.__input.read()
            # write output frame
            self.__output.write_frame(frame)
            # update selected filters
            for id, params in self._read_commands():
                if params is not None:
                    try:
                        filter = self.__filters[id]
                        filter.update(**params)
                        selected[id] = filter
                    except:
                        self.__logger.exception("update filter {} failed".format(id))    
                else:
                    selected.pop(id, None) 
            # process filters
            for id, filter in selected.items():
                try:
                    self.__output.write_filter(
                        id, filter.region(), filter.process(frame))
                except:
                    self.__logger.exception("process filter {} failed".format(id))
            # output flush
            self.__output.flush(observer)
    
    # -----------------------------------------------------------------------------------
    # write commands
    # -----------------------------------------------------------------------------------
    def set_filter(self, id, **kargs):
        self.__select.append((id, kargs))
        return self
    def clr_filter(self, id):
        self.__select.append((id, None))
        return self
    def set_filters(self, **kargs):
        for id in self.__filters:
            self.__select.append((id, kargs))
        return self
    def clr_filters(self):
        for id in self.__filters:
            self.__select.append((id, None))
        return self
    
    # -----------------------------------------------------------------------------------
    # read commands
    # -----------------------------------------------------------------------------------
    def _read_commands(self):
        # get actions
        select = self.__select
        # clear container 
        self.__select = []
        return select

    # -----------------------------------------------------------------------------------
    # load filters
    # -----------------------------------------------------------------------------------
    def _load_filters(self, conf):
        filters = {}
        for filter_key, filter_conf in conf['filters'].items():
            try:
                filters[filter_key] = VisionConfigFilter.Build(
                    filter_conf['type'],
                    filter_conf.get('conf'))
            except Exception as ex:
                self.__logger.exception(ex)
        return filters

# #################################################################################################
# -------------------------------------------------------------------------------------------------
# End
# -------------------------------------------------------------------------------------------------
# #################################################################################################
