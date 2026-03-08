import argparse
import cv2
import os
from src.pipeline.end2end import run_pipeline


def resize_for_display(image, max_width=1200):
    """
    Resize image if it is larger than screen width
    """
    h, w = image.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    return image


def main():

    parser = argparse.ArgumentParser(
        description="End-to-End Scene Text Detection and Recognition"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result image to results/"
    )

    args = parser.parse_args()

    image_path = args.image

    # kiểm tra ảnh tồn tại
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    print(f"Processing image: {image_path}")

    # chạy pipeline OCR
    output = run_pipeline(image_path)

    # resize để hiển thị đầy đủ
    display_image = resize_for_display(output)

    # tạo cửa sổ có thể resize
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", display_image)

    # lưu kết quả nếu cần
    if args.save:

        os.makedirs("results", exist_ok=True)

        filename = os.path.basename(image_path)
        save_path = os.path.join("results", f"result_{filename}")

        cv2.imwrite(save_path, output)

        print(f"✅ Result saved to: {save_path}")

    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()