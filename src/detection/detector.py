import easyocr

class TextDetector:
    """
    Text detection module using EasyOCR (CRAFT)
    """

    def __init__(self, lang=["en"], gpu=True):

        print("Loading text detection model...")

        self.reader = easyocr.Reader(lang, gpu=gpu)

    def detect(self, image):

        results = self.reader.readtext(image)

        boxes = []
        texts = []
        confs = []

        for bbox, text, conf in results:

            boxes.append(bbox)
            texts.append(text)
            confs.append(conf)

        return boxes, texts, confs