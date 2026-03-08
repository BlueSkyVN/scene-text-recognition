from src.evaluation.evaluate_icdar import evaluate_dataset

image_dir = "data/icdar2015/train_images"
gt_dir = "data/icdar2015/train_gt"

evaluate_dataset(image_dir, gt_dir)