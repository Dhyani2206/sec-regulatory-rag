from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel, Field

class CompanyOption(BaseModel):
    ticker: str
    available_forms: Dict[str, List[int]] = Field(default_factory=dict)

class OptionsResponse(BaseModel):
    companies: List[CompanyOption] = Field(default_factory=list)