import cv2
import numpy as np
from .icdar_parser import read_icdar_gt


def visualize_gt(image_path, gt_path):

    image = cv2.imread(image_path)

    boxes, texts = read_icdar_gt(gt_path)

    for box, text in zip(boxes, texts):

        pts = np.array(box).astype(int)

        if text == "###":
            color = (0,0,255)
        else:
            color = (0,255,0)

        cv2.polylines(image, [pts], True, color, 2)

        x = pts[0][0]
        y = pts[0][1] - 5

        cv2.putText(
            image,
            text,
            (x,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    return image