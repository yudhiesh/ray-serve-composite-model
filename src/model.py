from typing import Tuple

import tensorflow as tf
from transformers import BertTokenizer
from schema import Category

from src.constants import MODEL_NAME, MODEL_PATH

class_mapping: dict = {0: "hoax", 1: "news"}


class Model:
    def __init__(self, model_path: str, model_name: str) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = self.__load_model()

    def predict(self, text: str) -> Tuple[float, Category]:
        inputs = self.tokenizer(text, return_tensors="tf")
        input_ids = inputs["input_ids"]
        breakpoint()
        output = self.model(input_ids)
        probability = output[0][0].numpy()
        category = (probability > 0.5).astype("int32")
        class_name = class_mapping.get(category)
        if class_name is None:
            raise Exception("Invalid probability score")
        return (probability, class_name)

    def __load_model(self):
        return tf.keras.models.load_model(self.model_path)


if __name__ == "__main__":

    model = Model(
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
    )
    output = model.predict(
        "This Republican senator thinks 'wokeness' is the cause of mass shootings"
    )
    print(output)
