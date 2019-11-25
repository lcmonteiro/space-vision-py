# ################################################################################################
# ------------------------------------------------------------------------------------------------
# File:   vision_model_text_east.py
# Author: Luis Monteiro
#
# Created on nov 17, 2019, 22:00 PM
# ------------------------------------------------------------------------------------------------
# ################################################################################################
#
# internal
from library           import VisionModel
from library.resources import vision_resource
# external
import cv2 as cv
#
# ################################################################################################
# ------------------------------------------------------------------------------------------------
# VisionModelTextEAST 
# ------------------------------------------------------------------------------------------------
# ################################################################################################
class VisionModelTextEAST(VisionModel):
    #
    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    #
    def __init__(self):
        super().__init__()
        # load network
        self.__net = cv.dnn.readNet(vision_resource('east-test'))
        # select output layers
        self.__output_layers = [
            'feature_fusion/Conv_7/Sigmoid',
            'feature_fusion/concat_3'
        ]
    #
    # -------------------------------------------------------------------------
    # process
    # -------------------------------------------------------------------------
    #
    def process(self, frame):
        # prepare input
        data = self.__prepare(frame)
        # run model
        data = self.__run(data)
        # decode results
        data = self.__decode(data)
        # Apply NMS
        boxes       = data[0]
        confidences = data[1]
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.5)
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            print(vertices)
            for j in range(4):
                cv.line(frame, tuple(vertices[j-1]), tuple(vertices[j]), (0, 255, 0), 1)
        # Display the frame
        cv.imshow('detect', frame)
        return []
        return data
    #
    # -------------------------------------------------------------------------
    # steps 1 - input preparation
    # -------------------------------------------------------------------------
    #         
    def __prepare(self, frame):
        # Create a 4D blob from frame.
        height = frame.shape[0]
        width  = frame.shape[1]
        return cv.dnn.blobFromImage(
            frame, 1.0, (width, height), (123.68, 116.78, 103.94), True, False)
    #
    # -------------------------------------------------------------------------
    # steps 2 - run model
    # -------------------------------------------------------------------------
    # 
    def __run(self, blob):
        self.__net.setInput(blob)
        # return scores, geometry 
        return self.__net.forward(self.__output_layers)[0:2]
    #
    # -------------------------------------------------------------------------
    # step 3 - decode result
    # -------------------------------------------------------------------------
    #
    def __decode(self, data, threshold=0.5):
        from math import sin, cos, pi
        scores, geometry = data
        detections  = []
        confidences = []
        height = scores.shape[2]
        width  = scores.shape[3]
        for y in range(0, height):
            # Extract data from scores
            sc_data = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            an_data = geometry[0][4][y]
            for x in range(0, width):
                score = sc_data[x]
                # If score is lower than threshold score, move to next x
                if(score < threshold):
                    continue
                # Calculate offset
                x_offset = x * 4.0
                y_offset = y * 4.0
                angle    = an_data[x]
                # Calculate cos and sin of angle
                a_cos = cos(angle)
                a_sin = sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]
                # Calculate offset
                offset = ([
                    x_offset + a_cos * x1_data[x] + a_sin * x2_data[x], 
                    y_offset - a_sin * x1_data[x] + a_cos * x2_data[x]
                ])
                # Find points for rectangle
                p1 = (-a_sin * h + offset[0], -a_cos * h + offset[1])
                p3 = (-a_cos * w + offset[0],  a_sin * w + offset[1])
                # calculate center
                center = (
                    0.5 * (p1[0] + p3[0]), 
                    0.5 * (p1[1] + p3[1])
                )
                detections.append((center, (w, h), -angle * 180.0 / pi))
                confidences.append(float(score))
        # Return detections and confidences
        return [detections, confidences]

# ################################################################################################
# ------------------------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------------------------
# ################################################################################################
