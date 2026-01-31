from fastapi import FastAPI

app = FastAPI(title="Vattenfall Imbalance Price Predictor")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Vattenfall ML Service"}
