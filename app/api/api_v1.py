from fastapi import APIRouter

from app.api.endpoints import gpt

api_router = APIRouter()
api_router.include_router(gpt.router, prefix='/gpt', tags=['gpt'])
