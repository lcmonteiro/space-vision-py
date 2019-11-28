# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   __init__.py
# Author: Luis Monteiro
#
# Created on nov 17, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################
from os.path    import dirname
# -----------------------------------------------------------------------------
# get vision resource
# -----------------------------------------------------------------------------
def resource_path(id):
    return {
        'text-detection-east'     : '{root}/frozen_east_text_detection.pb'
    }[id].format(root=dirname(__file__))
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################
if __name__ == '__main__':
    import cv2      as cv
    import torch
    from torchsummary import summary

    model = torch.load(vision_model_resource('test-recognition-resnet'))['state_dict']
    print(model)
    for param_tensor in model:
        print(param_tensor, "\t", model[param_tensor].size())

    
    #torch.onnx.export()
    
    exit(0)
    # load network
    #net = cv.dnn.readNetFromTorch(vision_model_resource('test-recognition-resnet'))

    help(net)
    # select output layers
    output_layers = [
        'feature_fusion/Conv_7/Sigmoid',
        'feature_fusion/concat_3'
    ]
