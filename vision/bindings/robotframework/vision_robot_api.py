###################################################################################################
# ---------------------------------------------------------------------------------------
# File:   vision_robot_api.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ---------------------------------------------------------------------------------------
###################################################################################################

# external
from re       import match
from time     import time, sleep 
from logging  import getLogger  as logger

# internal
from vision.library import VisionDetector
from vision.library import VisionDatabase

###################################################################################################
# ---------------------------------------------------------------------------------------
# Vision API
# ---------------------------------------------------------------------------------------
###################################################################################################
class VisionApi(object):
    # -----------------------------------------------------------------------------------
    #   initialization
    # -----------------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        # init variables
        self.__logger   = logger()
        self.__detector = VisionDetector(config)
        self.__database = VisionDatabase()

        self.__functions = {}
        self.__default   = lambda k, x : {}
        self.__clients   = clients
        # start all clients
        for c in self.__clients:
            #set callbacks
            c.set_callback(self.__process)
            c.start()

    # -----------------------------------------------------------------------------------
    #   process
    # -----------------------------------------------------------------------------------
    def __process(self, key, data):
        try:
            self.__save(self.__functions[key](key, data))
        except:
            self.__save(self.__default(key, data))

    # -----------------------------------------------------------------------------------
    #   save on data base
    # -----------------------------------------------------------------------------------
    def __save(self, data):
        for key, value in data.items():
            if key not in self.__data_base or value != self.__data_base[key][0]:
                # log information
                self.__log.info('cache[{}]={}'.format(key, (value, time())))
                # save information
                self.__data_base[key] = (value, time())

    # -----------------------------------------------------------------------------------
    #   clear data base
    # -----------------------------------------------------------------------------------
    def clear_cache(self):  
        self.__data_base = {}

    # -----------------------------------------------------------------------------------
    #   get variable
    # -----------------------------------------------------------------------------------
    def get_variable(self, varname, gap_inf, gap_sup, value_inf, value_sup):
        time_start = float(time()) + int(gap_inf) if gap_inf else 0.0
        time_end   = float(time()) + int(gap_sup)
        var_start  = float(value_inf)
        var_end    = float(value_sup)
        # log information
        self.__log.info('time_beg={} time_end={} value_beg={} value_end={}'.format(
            time_start, time_end, var_start, var_end))
        # wait and validation 
        while  time() < time_end:
            # log debug
            #self.__log.debug('time={}'.format(time()))
            # check cache
            if varname in self.__data_base:
                # validation
                if self.__data_base[varname][1] <= time_start:
                    continue
                if float(self.__data_base[varname][0]) < var_start:
                     continue
                if float(self.__data_base[varname][0]) > var_end:
                     continue
                return self.__data_base[varname][0]
            sleep(0.1)
        # log error
        self.__log.error('time_beg={} time_end={} value_beg={} value_end={}'.format(
            time_start, time_end, var_start, var_end))
        # raise an exception  
        raise ValueError('not found(var=%s)'%(varname))

###################################################################################################
# ---------------------------------------------------------------------------------------
#   Test
# ---------------------------------------------------------------------------------------
###################################################################################################
if __name__ == '__main__':
    dlt= VisionApi()
    try: 
        print(dlt.get_variable("abc", (0, 1), (5000, float('inf'))))
    except Exception as e:
        print(str(e))

###################################################################################################
# ---------------------------------------------------------------------------------------
#   End
# ---------------------------------------------------------------------------------------
###################################################################################################