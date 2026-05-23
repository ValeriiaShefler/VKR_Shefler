from fastapi import APIRouter, HTTPException, Response
from ..models.isa_core import standard_atmosphere
import pandas as pd
import json
from typing import Literal

router = APIRouter(prefix="/export", tags=["Экспорт"])

def generate_xml(data):
    """Простая генерация XML через строки (без внешних библиотек)"""
    xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<isa_profile>']
    for row in data:
        xml_lines.append('  <point>')
        for key, value in row.items():
            xml_lines.append(f'    <{key}>{value}</{key}>')
        xml_lines.append('  </point>')
    xml_lines.append('</isa_profile>')
    return '\n'.join(xml_lines)

@router.get("/full_profile")
async def export_full_profile(format: Literal["csv", "json", "xml", "txt"] = "csv"):
    heights = list(range(0, 20001, 100))
    data = []
    for h in heights:
        T, P, rho, a = standard_atmosphere(h)
        data.append({"h": h, "T": round(T, 2), "P": round(P, 2), "ρ": round(rho, 4), "a": round(a, 2)})
    
    if format == "csv":
        df = pd.DataFrame(data)
        content = df.to_csv(index=False)
        media_type = "text/csv"
        filename = "isa_profile.csv"
    elif format == "json":
        content = json.dumps(data, indent=2, ensure_ascii=False)
        media_type = "application/json"
        filename = "isa_profile.json"
    elif format == "xml":
        content = generate_xml(data)
        media_type = "application/xml"
        filename = "isa_profile.xml"
    elif format == "txt":
        df = pd.DataFrame(data)
        content = df.to_csv(sep='\t', index=False)
        media_type = "text/plain"
        filename = "isa_profile.txt"
    else:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")
    
    return Response(content=content, media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"})