import os
import cv2
import numpy as np
import warnings

from src.utils.icdar_parser import read_icdar_gt
from src.detection.detector import TextDetector


warnings.filterwarnings("ignore")


def polygon_to_bbox(polygon):
    """
    Convert polygon (4x2) to bounding box [x1,y1,x2,y2]
    """

    xs = polygon[:, 0]
    ys = polygon[:, 1]

    x_min = xs.min()
    y_min = ys.min()
    x_max = xs.max()
    y_max = ys.max()

    return [x_min, y_min, x_max, y_max]


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union between two bounding boxes
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area

    if union == 0:
        return 0

    return inter_area / union


def evaluate_dataset(image_dir, gt_dir, iou_threshold=0.5):
    """
    Evaluate OCR detection results on ICDAR dataset
    """

    print("Loading text detection model...")
    detector = TextDetector()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    images = sorted(os.listdir(image_dir))

    print("\nStarting evaluation...\n")

    for idx, img_name in enumerate(images):

        image_path = os.path.join(image_dir, img_name)

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        gt_name = "gt_" + os.path.splitext(img_name)[0] + ".txt"
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            print(f"Skipping {img_name} (GT not found)")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {img_name}")
            continue

        # -------- PREDICTION --------

        pred_polygons, _, _ = detector.detect(image)

        pred_boxes = []

        for poly in pred_polygons:

            poly = np.array(poly)
            bbox = polygon_to_bbox(poly)
            pred_boxes.append(bbox)

        # -------- GROUND TRUTH --------

        gt_polygons, texts = read_icdar_gt(gt_path)

        gt_boxes = []

        for polygon, text in zip(gt_polygons, texts):

            # Ignore region
            if text == "###":
                continue

            bbox = polygon_to_bbox(np.array(polygon))
            gt_boxes.append(bbox)

        # -------- MATCHING --------

        matched_gt = set()

        tp = 0
        fp = 0

        for pred in pred_boxes:

            best_iou = 0
            best_gt = -1

            for i, gt in enumerate(gt_boxes):

                iou = compute_iou(pred, gt)

                if iou > best_iou:
                    best_iou = iou
                    best_gt = i

            if best_iou >= iou_threshold and best_gt not in matched_gt:

                tp += 1
                matched_gt.add(best_gt)

            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"[{idx+1}/{len(images)}] {img_name} | TP:{tp} FP:{fp} FN:{fn}"
        )

    # -------- FINAL METRICS --------

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n==============================")
    print(" FINAL EVALUATION RESULTS")
    print("==============================")

    print(f"Total TP : {total_tp}")
    print(f"Total FP : {total_fp}")
    print(f"Total FN : {total_fn}")

    print("\nMetrics:")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nEvaluation completed.")