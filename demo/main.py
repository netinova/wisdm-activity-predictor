from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sensor_batch import SensorBatch
from sklearn.pipeline import Pipeline
from typing import Literal
import os
import joblib
import pandas as pd
from transformers import RawFeatureTransformer

top = os.path.dirname(__file__)


app = FastAPI()

model_phone: Pipeline = joblib.load(f"{top}/phone_train.pkl")
model_watch: Pipeline = joblib.load(f"{top}/watch_train.pkl")

ACTIVITIES = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Sitting",
    "E": "Standing",
    "F": "Typing",
    "G": "Brushing Teeth",
    "H": "Eating Soup",
    "I": "Eating Chips",
    "J": "Eating Pasta",
    "K": "Drinking from Cup",
    "L": "Eating Sandwich",
    "M": "Kicking (Soccer Ball)",
    "O": "Playing Catch w/Tennis Ball",
    "P": "Dribbling (Basketball)",
    "Q": "Writing",
    "R": "Clapping",
    "S": "Folding Clothes",
}


@app.post("/predict")
async def predict(batch: SensorBatch, mode: Literal["pocket", "hand"] = Query()):
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
                entry.timeStamp,
                entry.x_accel,
                entry.y_accel,
                entry.z_accel,
                entry.x_gyro,
                entry.y_gyro,
                entry.z_gyro,
            ]
            for entry in batch.data
        ],
        columns=columns,
    )

    df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit="ns")

    if mode == "pocket":

        prediction = model_phone.predict(df)[0]

        try:
            probabilities = model_phone.predict_proba(df)
            confidence = float(probabilities.max())
        except:
            confidence = None

        return {"prediction": ACTIVITIES[str(prediction)], "confidence": confidence}
    elif mode == "hand":
        prediction = model_watch.predict(df)[0]

        try:
            probabilities = model_watch.predict_proba(df)
            confidence = float(probabilities.max())
        except:
            confidence = None
        return {"prediction": ACTIVITIES[str(prediction)], "confidence": confidence}


app.mount("/", StaticFiles(directory=f"{top}/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True)
