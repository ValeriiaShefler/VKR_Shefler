from pydantic import BaseModel
from typing import Optional, Dict, Any

class ISAParams(BaseModel):
    height_m: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float
    source: str = "ГОСТ 4401-81 / ISO 2533"

class AIPrediction(BaseModel):
    height_m: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float
    model_type: str = "MLPRegressor"

class VerificationMetrics(BaseModel):
    parameter: str
    mae: float
    rmse: float
    r2: float

class ExportResponse(BaseModel):
    message: str
    format: str
    data: Optional[Any] = None