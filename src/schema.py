from pydantic import BaseModel
from enum import Enum


class Category(str, Enum):
    hoax = "hoax"
    news = "news"


class ModelInput(BaseModel):
    text: str


class ModelOutput(BaseModel):
    score: float
    news_type: Category


class CompositeModelOutput(ModelOutput):
    text: str
