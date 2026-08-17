from pydantic import BaseModel


class SensorEntry(BaseModel):
    timeStamp: int
    x_accel: float
    y_accel: float
    z_accel: float
    x_gyro: float
    y_gyro: float
    z_gyro: float


class SensorBatch(BaseModel):
    data: list[SensorEntry]
