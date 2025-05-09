import pickle
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
# Создаем интрефейс

app = FastAPI()
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionInput(BaseModel):
    total_square: float
    rooms: int
    floor: int

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/get_predict")
def get_predict(total_square: float = Query(...), rooms: int = Query(...), floor: int = Query(...)):
    X = np.array([[total_square, rooms, floor]])
    prediction = model.predict(X)[0][0]
    return {"prediction": round(prediction,2)}


@app.post("/post_predict")
def post_predict(data: PredictionInput):
    try:
        model_data = {
            "total_square": data.total_square,
            "rooms": data.rooms,
            "floor": data.floor
        }

        df = pd.DataFrame([model_data])[model.feature_names_in_]
        prediction = model.predict(df)[0][0]
        print(prediction)
        return {"result": round(prediction,2) }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



        

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
