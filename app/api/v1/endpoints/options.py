from __future__ import annotations
from fastapi import APIRouter
from app.api.v1.schemas.options import OptionsResponse
from app.services.options_service import get_available_options

router = APIRouter()

@router.get("/options", response_model=OptionsResponse, summary="Available filing scope options")
def options_endpoint() -> OptionsResponse:
    return get_available_options()