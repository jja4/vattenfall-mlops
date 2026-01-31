import os
import requests
import pandas as pd
from datetime import datetime, timedelta

class FingridClient:
    BASE_URL = "https://data.fingrid.fi/api"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"x-api-key": self.api_key}

    def get_dataset(self, dataset_id: int, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        pass
