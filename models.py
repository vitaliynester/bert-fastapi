from pydantic import BaseModel


class RequestModel(BaseModel):
    text1: str
    text2: str
