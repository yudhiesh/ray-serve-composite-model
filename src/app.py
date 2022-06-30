from fastapi import FastAPI, HTTPException, status
import ray
from ray import serve

from src.constants import MODEL_NAME, MODEL_PATH
from src.model import Model
from src.schema import ModelInput, ModelOutput


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
class NLPClassifier:
    def __init__(self) -> None:
        self.model = Model(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME,
        )

    @app.post(
        "/predict",
        response_model=ModelOutput,
        status_code=status.HTTP_200_OK,
    )
    async def predict(self, payload: ModelInput):
        text = payload.text
        output = self.model.predict(text=text)
        if not output:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error getting prediction",
            )
        probability, class_name = output
        return ModelOutput(
            score=probability,
            news_type=class_name,
        )


NLPClassifier.deploy()
