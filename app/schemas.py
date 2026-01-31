from pydantic import BaseModel

class PredictionRequest(BaseModel):
    pass

class PredictionResponse(BaseModel):
    price_prediction: float
    unit: str = "EUR/MWh"
