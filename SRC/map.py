
import cv2
from config import *


def draw_map(frame, map_img, W, H):
    # Get Image shape
    rows,cols,channels = map_img.shape

    # Create overlay: SRC1, Alpha, Src2, Beta, Gamma
    overlay=cv2.addWeighted(frame[y_offset:y_offset+rows, x_offset:x_offset+cols], \
        map_alpha, map_img, map_beta, map_gamma)

    frame[y_offset:y_offset+rows, x_offset:x_offset+cols ] = overlay