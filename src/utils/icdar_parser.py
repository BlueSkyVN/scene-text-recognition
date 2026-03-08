import numpy as np


def read_icdar_gt(gt_path):

    boxes = []
    texts = []

    with open(gt_path, "r", encoding="utf-8-sig") as f:

        for line in f.readlines():

            parts = line.strip().split(",")

            coords = list(map(int, parts[:8]))

            text = ",".join(parts[8:])

            points = np.array(coords).reshape(4,2)

            boxes.append(points)
            texts.append(text)

    return boxes, texts