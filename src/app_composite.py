from typing import List, Tuple
from fastapi import FastAPI, status
from nltk.tokenize import sent_tokenize
import ray
from ray import serve
import tensorflow as tf
from transformers import AutoTokenizer
import asyncio

from constants import MODEL_NAME, MODEL_PATH
from src.model import class_mapping
from src.schema import Category, CompositeModelOutput, ModelInput


@serve.deployment(version="v1")
class AutoTokenizerDeployment:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def __call__(
        self,
        sentence: str,
    ) -> tf.Tensor:
        inputs = self.tokenizer(sentence, return_tensors="tf")
        input_ids = inputs.get("input_ids")
        return input_ids


@serve.deployment(version="v0")
class SplitSentencesDeployment:
    async def __call__(self, text: List[str]) -> List[str]:
        return sent_tokenize(text)


@serve.deployment(num_replicas=4, version="v0")
class ModelDeployment:
    def __init__(self, model_path: str) -> None:
        self._model = tf.keras.models.load_model(model_path)

    async def __call__(self, preprocessed) -> Tuple[float, Category]:
        output = self._model(preprocessed)
        probability = output[0][0].numpy()
        category = (probability > 0.5).astype("int32")
        class_name = class_mapping.get(category)
        if class_name is None:
            raise Exception("Invalid probability score")
        return (probability, class_name)


app = FastAPI()
ray.init(address="auto", namespace="nlp")
serve.start(
    detached=True,
    http_options={
        "host": "0.0.0.0",
        "port": 5050,
    },
)


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class NLPClassifierComposite:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizerDeployment.get_handle(sync=False)
        self.sentence_splitter = SplitSentencesDeployment.get_handle(sync=False)
        self.model = ModelDeployment.get_handle(sync=False)

    @app.post(
        "/predict",
        # response_model=List[CompositeModelOutput],
        # throws an error due to bug in Ray
        status_code=status.HTTP_200_OK,
    )
    async def predict(
        self,
        payload: ModelInput,
    ) -> List[CompositeModelOutput]:
        text = payload.text
        sentences = await (await self.sentence_splitter.remote(text))
        tokenized_sentence_coroutines = [
            self.tokenizer.remote(sentence) for sentence in sentences
        ]
        tokenized_sentence_refs = await asyncio.gather(*tokenized_sentence_coroutines)
        # 1. Get list of coroutines
        output_coroutines = [
            self.model.remote(tokenized_sentence)
            for tokenized_sentence in tokenized_sentence_refs
        ]
        # output_coroutines = [<coroutine object RayServeHandle.remote at 0x7f9e63af7e40>, <coroutine object RayServeHandle.remote at 0x7f9e63d47340>]
        # 2. Run coroutines async
        output_refs = await asyncio.gather(*output_coroutines)
        # output_refs = [ObjectRef(64409601d02d0644559bb50003067d222616a8630100000002000000), ObjectRef(bd08493e267536708462fa7921045c7bd71655940100000002000000)]
        # 3. Run the values of the outputs
        # Could use ray.get() here but that would block the execution thread
        predictions = await asyncio.gather(*output_refs)
        return [
            {
                "sentence": sentence,
                "news_type": class_name,
                "score": float(
                    probability,
                ),
            }
            for sentence, (probability, class_name) in zip(sentences, predictions)
        ]


print("Deploying model...")
AutoTokenizerDeployment.deploy(MODEL_NAME)
SplitSentencesDeployment.deploy()
ModelDeployment.deploy(MODEL_PATH)
NLPClassifierComposite.deploy()
print("Deployment done!")
