from datetime import datetime
from beam import endpoint, Volume, Image
import joblib
import numpy as np


def load_model():
    try:
        model = joblib.load("./model.pkl")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_simple_model():
    try:
        model = joblib.load("./simplemodel.pkl")
        print("Simple model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading simple model: {e}")
        return None

@endpoint(
    cpu=1,
    memory=1,
    keep_warm_seconds=0,
    name="mlstockprice",
    image=Image(python_packages=["joblib","scikit-learn==1.5.2"]),
    on_start=load_model,
)
def mlstockprice(context, **inputs):
    model = context.on_start_value

    date = datetime.strptime(inputs["date"], "%d/%m/%Y")
    year = date.year
    month = date.month
    day = date.day

    X= [[inputs["open"], inputs["high"], inputs["low"], year, month, day]]

    result = model.predict(X)
    print(f"Prediction result: {result}")
    return {"result": result.tolist()}


@endpoint(
    cpu=0.8,
    memory=1,
    keep_warm_seconds=0,
    name="mlstockpricesimple",
    image=Image(python_packages=["joblib","numpy"]),
    on_start=load_simple_model,
)
def mlstockprice_simple(context, **inputs):
    model = context.on_start_value

    date = datetime.strptime(inputs["date"], "%d/%m/%Y")
    year = date.year
    month = date.month
    day = date.day

    X= [[inputs["open"], inputs["high"], inputs["low"], year, month, day]]

    result = np.dot(X, model['coef']) + model['intercept']
    print(f"Prediction result: {result}")
    return {"result": result.tolist()}

if __name__ == "__main__":
    mlstockprice_simple.local