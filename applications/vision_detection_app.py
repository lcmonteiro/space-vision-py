#
# ------------------------------------------------------------------------------------------------
# File:   vision_detection_app.py
# Author: Luis Monteiro
#
# Created on oct 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# 
# extern
#
from yaml import safe_load as loader
#
# intern
#
from library.vision_detection import VisionDetection
#
# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
#
def main(args):
    #
    # ----------------------------------------------------
    # init and load filters 
    # ----------------------------------------------------
    #                
    vd = VisionDetection(loader(open(args['config'])))
    #
    # ----------------------------------------------------
    # configure filter
    # ----------------------------------------------------
    #
    vd.set_filters()
    #
    # ----------------------------------------------------
    # run detection
    # ----------------------------------------------------
    #
    @vd.serve
    def process(id, result):
        print('result::id={}'.format(id))
#
# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
#
if __name__ == '__main__':
    from argparse import ArgumentParser
    from logging  import basicConfig     as config_logger
    from logging  import getLogger       as logger
    from logging  import DEBUG           as LEVEL
    from sys      import stdout
    #
    # ---------------------------------------------------------------
    # parse parameters
    # ---------------------------------------------------------------
    #
    parser = ArgumentParser()
    # configuration path
    parser.add_argument('--config', '-c', 
        type    = str, 
        default = 'vision_detection_app.yaml',
        help    = 'configuration file path')
    args = parser.parse_args()
    # 
    # ---------------------------------------------------------------
    # log configuration
    # ---------------------------------------------------------------
    #
    config_logger(
        stream   = stdout,
        filemode = 'w',
        level    = LEVEL, 
        #filename= 'vision_detection_app.log', 
        format   = '[%(asctime)s] [%(levelname)-10s] [%(funcName)s] %(message)s')
    #
    # ---------------------------------------------------------------
    # main 
    # ---------------------------------------------------------------
    #
    try:
        exit(main(vars(args)))
    except Exception as e:
        logger().exception(e)
        exit(-1)
#
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
#