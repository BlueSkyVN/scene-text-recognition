import cv2
import os
from src.utils.visualize_gt import visualize_gt

image = "data/icdar2015/train_images/img_1.jpg"
gt = "data/icdar2015/train_gt/gt_img_1.txt"

output = visualize_gt(image, gt)

# tạo thư mục results nếu chưa tồn tại
os.makedirs("results", exist_ok=True)

# lưu ảnh
cv2.imwrite("results/gt_img_1_visualized.jpg", output)

# hiển thị ảnh
cv2.imshow("GT", output)
cv2.waitKey(0)
cv2.destroyAllWindows()