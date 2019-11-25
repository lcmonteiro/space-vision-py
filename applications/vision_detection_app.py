# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   vision_detection_app.py
# Author: Luis Monteiro
#
# Created on nov 8, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################ 
#
# extern
#
from yaml    import safe_load as loader
from logging import getLogger as logger
#
# intern
#
from library         import VisionDetection
from library.inputs  import VisionInputCamera
from library.inputs  import VisionInputFilesystem
from library.outputs import VisionOutputWindow
# #############################################################################
# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
# #############################################################################
def main(args):
    print(args)
    #
    # ----------------------------------------------------
    # init and load filters 
    # ----------------------------------------------------
    #                
    vision_detection = VisionDetection(
        # configuration
        loader(open(args['config'])), 
        # input options
        {
            'camera'     : VisionInputCamera,
            'filesystem' : VisionInputFilesystem
        }[args['input']](args['src']),
        # output options
        {
            'window'    : VisionOutputWindow
        }[args['output']](args['dst'])
    )
    #
    # ----------------------------------------------------
    # configure filter
    # ----------------------------------------------------
    #
    vision_detection.set_filters()
    #
    # ----------------------------------------------------
    # run detection
    # ----------------------------------------------------
    #
    @vision_detection.serve
    def process(id, result):
        logger().info('filter={} label={}]'.format(
            id, result.label()
        ))
# ############################################################################
# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
# ############################################################################
if __name__ == '__main__':
    from argparse import ArgumentParser
    from logging  import basicConfig     as config_logger
    from logging  import DEBUG           as LEVEL
    from sys      import stdout
    from os.path  import abspath, dirname
    import seaborn as sns
    sns.set_palette("hls")
    #
    # ---------------------------------------------------------------
    # parse parameters
    # ---------------------------------------------------------------
    #
    parser = ArgumentParser()
    # configuration path
    parser.add_argument('--config', '-c', 
        type    = str, 
        default = '%s/vision_detection_app.yaml'%(dirname(abspath(__file__))),
        help    = 'configuration file path')
    # input options
    parser.add_argument('--input', '-i', 
        type    = str, 
        default = 'camera',
        choices =['camera', 'filesystem'],
        help    = 'input option')
    # output options
    parser.add_argument('--output', '-o', 
        type    = str, 
        default = 'window',
        choices =['window'],
        help    = 'output option')
    parser.add_argument('src', 
        default = '0',
        nargs   = '?',
        help    ='source id')
    parser.add_argument('dst', 
        default = 'vision detection',
        nargs   = '?',
        help    ='destination id')
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
        format   = 
            '[%(asctime)s] '
            '[%(levelname)-10s] '
            '[%(funcName)s] %(message)s')
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
    except KeyboardInterrupt:
        exit(0)
# #################################################################################################
# -------------------------------------------------------------------------------------------------
# End
# -------------------------------------------------------------------------------------------------
# #################################################################################################