import cv2
import argparse
import numpy as np
import tensorflow as tf
from collections import deque
from matplotlib import pyplot as plt


from Homography.utils.vizualization import visualize
from Homography.models.keras_models import DeepHomoModel
from Homography.utils.vizualization import rgb_template_to_coord_conv_template
from Homography.utils.homography import compute_homography, warp_image
from Homography.utils.image import torch_img_to_np_img, np_img_to_torch_img
from Homography.utils.utils import to_torch

from utils import *
from config import *
from map import *


# Declare Arguments
ap = argparse.ArgumentParser()
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--GPU", action="store_true")
ap.add_argument("--write", action="store_true")


# read the class names
with open('Model/football.txt', 'r') as f:
    football_classes = f.read().split('\n')


# final class names
class_names = [name.split(',')[0] for name in football_classes]

pts = deque(maxlen=num_tracked_points)

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


# Read the Deep Homography Estimation Model
deep_homo_model = DeepHomoModel()
WEIGHTS_PATH = (DeepHomoModel_weights)
checkpoints = tf.keras.utils.get_file(
                WEIGHTS_NAME, WEIGHTS_PATH, WEIGHTS_TOTAR,)
deep_homo_model.load_weights(checkpoints)



# Map Settings
template = cv2.imread(map_location)
template = cv2.resize(cv2.cvtColor(template, cv2.COLOR_BGR2RGB), (1280,720))/255.

map_template = cv2.imread(map_location)
map_template = cv2.resize(map_template, (1280,720))
template = rgb_template_to_coord_conv_template(template)



# capture the video
cap = cv2.VideoCapture(input_video)

if(cap.isOpened()):
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`


if(write):
    vid_writer_map = cv2.VideoWriter(output_map, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(np.shape(template)[1]),round(np.shape(template)[0])))
    vid_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    vid_writer_warp = cv2.VideoWriter(output_warp, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280, 720))


while cv2.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:

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

    # Field localaization
    corners = deep_homo_model(frame)
    pred_homo = compute_homography(corners)[0]
    pred_warp = warp_image(np_img_to_torch_img(template),to_torch(np.linalg.inv(pred_homo)),method='torch')
    pred_warp = torch_img_to_np_img(pred_warp[0])

    #visualize(image=frame,warped_template=cv2.resize(pred_warp, (1280,720)))
    
    # Remove the bounding boxes with low confidence
    map = postprocess(frame, outs, class_names, pts, np.linalg.inv(pred_homo), map_template.copy())

    # Show Image
    cv2.imshow("frame", cv2.resize(frame.copy(), (resized_width, resized_height)))

    if(write):
        vid_writer.write(frame.astype(np.uint8))
        vid_writer_map.write(map.astype(np.uint8))
        vid_writer_warp.write(cv2.resize(pred_warp, (1280,720)).astype(np.uint8))


cv2.destroyAllWindows()

