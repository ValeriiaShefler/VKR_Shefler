"""
Роутер для работы с датасетами.
"""
from fastapi import APIRouter, HTTPException
from ..data_loader import list_datasets, get_dataset_info

router = APIRouter(prefix="/datasets", tags=["Датасеты"])


@router.get("/list")
async def get_datasets():
    """Список всех доступных датасетов."""
    return list_datasets()


@router.get("/info/{name}")
async def get_info(name: str):
    """Метаданные конкретного датасета."""
    try:
        return get_dataset_info(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))