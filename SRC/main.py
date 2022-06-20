import cv2
import numpy as np
from collections import deque
import argparse

from Yolo_utils import *
from config import *
from map import *

# read the class names
with open('Model/football.txt', 'r') as f:
    football_classes = f.read().split('\n')

# final class names
class_names = [name.split(',')[0] for name in football_classes]

pts = deque(maxlen=num_tracked_points)

# Declare Arguments
ap = argparse.ArgumentParser()
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--GPU", action="store_true")
ap.add_argument("--write", action="store_true")

# Extract Arguments
args = ap.parse_args()
verbose = args.verbose
write = args.write
GPU = args.GPU

# Read Net and set backend Target
net = cv2.dnn.readNetFromDarknet(model_configurations, model_weights)


# Define Backend for OpenCV
if(GPU):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Read Map Image
map_img = cv2.imread(map_location)
map_img = cv2.resize(map_img, (int(map_img.shape[1] * scale_percent / 100), int(map_img.shape[0] * scale_percent / 100)))


# capture the video
cap = cv2.VideoCapture(input_video)

if(cap.isOpened()):
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

if(verbose):
    print(f"Frame width is: {frame_width},  Frame height {frame_height} .")
    print(f"Map width is: {map_img.shape[1]},  Map height {map_img.shape[0]} .")

map_width = map_img.shape[1]
map_heigth = map_img.shape[0]

if(write):
    vid_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cv2.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        
        if(write):
            print("Done processing... Output file is stored as ", output_video)

        # Release device
        cv2.waitKey(3000)
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # OpenCV Draw Map
    draw_map(frame, map_img, int(frame_width), int(frame_height))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, class_names, pts)

    # Show Image
    cv2.imshow("frame", cv2.resize(frame.copy(), (resized_width, resized_height)))

    # Write the frame with the detection boxes
    if(write):
        vid_writer.write(frame.astype(np.uint8))

cv2.destroyAllWindows()

