class TextRecognizer:
    """
    Recognition post-processing module
    """

    def __init__(self, conf_threshold=0.3):

        self.conf_threshold = conf_threshold

    def recognize(self, texts, confs):

        final_texts = []

        for text, conf in zip(texts, confs):

            if conf >= self.conf_threshold:
                final_texts.append(text)

            else:
                final_texts.append("")

        return final_texts