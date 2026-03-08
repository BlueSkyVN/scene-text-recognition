import cv2

from src.detection.detector import TextDetector
from src.recognition.recognizer import TextRecognizer
from src.utils.visualization import draw_ocr_results


class SceneTextPipeline:

    def __init__(self):

        self.detector = TextDetector()

        self.recognizer = TextRecognizer()

    def process(self, image):

        # Step 1: Detection
        boxes, texts, confs = self.detector.detect(image)

        # Step 2: Recognition filtering
        texts = self.recognizer.recognize(texts, confs)

        # Step 3: Visualization
        output = draw_ocr_results(image, boxes, texts)

        return output


pipeline = SceneTextPipeline()


def run_pipeline(image_path):

    image = cv2.imread(image_path)

    output = pipeline.process(image)

    return output