import cv2
import numpy as np

from config import *


def calculate_centroid(left, top, right, bottom):
    return (int((left+right)/2), int((top+bottom)/2))


# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom, class_names, pts):

    # Draw a bounding box.
    if(class_names[classId] == 'Ball'):
        thickness = 2
    else:
        thickness = 1
    cv2.rectangle(frame, (left, top), (right, bottom), COLORS[str(classId)], thickness)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if class_names:
        assert(classId < len(class_names))
        label = '%s:%s' % (class_names[classId], label)
        label = '%s' % (class_names[classId])

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    center = calculate_centroid(left, top, right, bottom)
    
    # TODO: Remore +122 across X axis for view purposes. 
    midmap = 0 # 122
    # TODO: Clean this section of code. It could be better
    if(label != 'Ball'):
        cv2.rectangle(frame, (left, top - round(1.0*labelSize[1])), (left + round(1.0*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # Draw Map of each player
        player_2D_x = (((center[0] - OldMin_x) * new_range_x) / old_range_x) + NewMin_x
        player_2D_y = (((center[1] - OldMin_y) * new_range_y) / old_range_y) + NewMin_y
        cv2.circle(frame, (int(player_2D_x) + x_offset+midmap, int(player_2D_y) + y_offset), 3, COLORS[str(classId)], 3)
    else:
        
        # Draw Circle on the Ball center
        cv2.circle(frame, center, 3, (0,0,255), 3)
        
        # Draw Map Position of the ball
        ball_2D_x = (((center[0] - OldMin_x) * new_range_x) / old_range_x) + NewMin_x
        ball_2D_y = (((center[1] - OldMin_y) * new_range_y) / old_range_y) + NewMin_y
        cv2.circle(frame, (int(ball_2D_x) + x_offset+midmap, int(ball_2D_y) + y_offset), 3, (0,0,255), 3)

        pts.appendleft(center)
    
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(num_tracked_points / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)



def postprocess(frame, outs, class_names, pts):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1] 
 
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])


    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height, class_names, pts)


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]