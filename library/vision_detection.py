#
# ------------------------------------------------------------------------------------------------
# File:   vision_detection.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
#
# external imports
from cv2      import waitKey    as wait
from logging  import getLogger  as logger
#
# internal imports
from library.inputs.vision_input_camera   import VisionInputCamera
from library.outputs.vision_output_window import VisionOutputWindow
from library.vision_config                import VisionFilterBuilder
#
# ------------------------------------------------------------------------------------------------
# VisionDetection 
# ------------------------------------------------------------------------------------------------
#
class VisionDetection:
    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------
    def __init__(self, config):
        # -------------------------------------------------
        # variables
        # -------------------------------------------------
        # command containers  
        self.__select = []
        # load logger
        self.__log = logger()
        # filter data base 
        self.__filters = self._load_filters(config)
        # load input 
        self.__input  = VisionInputCamera()
        # load output
        self.__output = VisionOutputWindow("")
    # ------------------------------------------------------------------------
    # serve
    # ------------------------------------------------------------------------
    def serve(self, callback):
        #
        # check vision input
        #
        if not self.__input.good():
            raise RuntimeError("vision input open fail ...")
        # 
        # image processing
        #
        selected = {}
        while wait(1) < 0:
            #
            # read frame
            #
            frame = self.__input.read()
            #        
            # update selected filters
            # 
            for id, params in self._read_commands():
                if params['level'] > 0 :
                    filter = self.__filters[id]
                    filter.update(params)
                    selected[id] = filter
                else:
                    selected.pop(id, None) 
            #
            # process filters
            #
            for id, filter in selected:
                try:
                    for result in filter.process(frame):
                        callback(id, result)
                except:
                    self.__log.warning("process filter {} failed".format(id))
            #
            # output
            #
            self.__output.write(frame)
    # ------------------------------------------------------------------------
    # write commands
    # ------------------------------------------------------------------------
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
        return self.__select
    #
    # ------------------------------------------------------------------------
    # load filters
    # ------------------------------------------------------------------------
    #
    def _load_filters(self, conf):
        filters = {}
        for filter_key, filter_conf in conf['filters'].items():
            try:
                filters[filter_key] = VisionFilterBuilder.Build(
                    filter_conf['type'],
                    filter_conf.get('conf'))
            except Exception as ex:
                self.__log.exception(ex)
        return filters
#
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
#
