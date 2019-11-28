# #################################################################################################
# -------------------------------------------------------------------------------------------------
# File:   filesystem_input.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# -------------------------------------------------------------------------------------------------
# #################################################################################################

# external
from cv2            import imread         
from glob           import glob        
# internal
from vision.library import VisionInput

# #############################################################################
# -----------------------------------------------------------------------------
# Filesystem Input 
# -----------------------------------------------------------------------------
# #############################################################################
class FilesystemInput(VisionInput):

    # -----------------------------------------------------
    # initialization
    # -----------------------------------------------------
    def __init__(self, source=0):
        def gen():
            paths = glob(source)
            if len(paths):
                for path in paths:
                    yield path
                yield from gen()
        # create generator
        self.__gen = gen()

    # -----------------------------------------------------
    # check status
    # -----------------------------------------------------
    def good(self):
        return True

    # -----------------------------------------------------
    # read frame
    # -----------------------------------------------------
    def read(self):
        frame = imread(next(self.__gen))
        return frame    

# #################################################################################################        
# -------------------------------------------------------------------------------------------------
# End
# -------------------------------------------------------------------------------------------------
# #################################################################################################