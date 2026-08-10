from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from sensor_batch import SensorBatch
from sklearn.pipeline import Pipeline
import os
import joblib
import pandas as pd
from transformers import RawFeatureTransformer

top = os.path.dirname(__file__)

app = FastAPI()

model: Pipeline = joblib.load(f"{top}/phone_train.pkl")


@app.post("/predict")
async def predict(batch: SensorBatch):

    columns = [
        "timeStamp",
        "x_accel",
        "y_accel",
        "z_accel",
        "x_gyro",
        "y_gyro",
        "z_gyro",
    ]

    df = pd.DataFrame(
        [
            [
                d["timeStamp"],
                d["x_accel"],
                d["y_accel"],
                d["z_accel"],
                d["x_gyro"],
                d["y_gyro"],
                d["z_gyro"],
            ]
            for d in batch.data
        ],
        columns=columns,
    )

    df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit="ns")

    prediction = model.predict(df)[0]
    return prediction


app.mount("/", StaticFiles(directory=f"{top}/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True)
