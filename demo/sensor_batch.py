from pydantic import BaseModel


class SensorBatch(BaseModel):
    data: list
