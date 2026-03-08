import cv2
import numpy as np


def draw_ocr_results(image, boxes, texts):

    output = image.copy()

    for bbox, text in zip(boxes, texts):

        pts = np.array(bbox).astype(int)

        cv2.polylines(output, [pts], True, (0,255,0), 2)

        x = pts[0][0]
        y = pts[0][1] - 10

        cv2.putText(
            output,
            text,
            (x,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    return output