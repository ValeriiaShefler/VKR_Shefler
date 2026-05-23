from fastapi import APIRouter, HTTPException
from ..models.isa_core import standard_atmosphere
from ..schemas.responses import ISAParams

router = APIRouter(prefix="/isa", tags=["International Standard Atmosphere"])

@router.get("/calc", response_model=ISAParams)
async def calculate_isa(height: float):
    if height < 0 or height > 20000:
        raise HTTPException(status_code=400, detail="Высота должна быть в диапазоне 0..20000 м")
    T, P, rho, a = standard_atmosphere(height)
    return ISAParams(
        height_m=height,
        temperature_K=round(T, 2),
        pressure_Pa=round(P, 2),
        density_kg_m3=round(rho, 4),
        speed_of_sound_m_s=round(a, 2),
        source="ГОСТ 4401-81 / ISO 2533"
    )