import cv2
from src.utils.visualize_gt import visualize_gt


image = "data/icdar2015/train_images/img_1.jpg"

gt = "data/icdar2015/train_gt/gt_img_1.txt"


output = visualize_gt(image, gt)

cv2.imshow("GT", output)

cv2.waitKey(0)