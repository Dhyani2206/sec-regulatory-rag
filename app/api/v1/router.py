from fastapi import APIRouter
from app.api.v1.endpoints.agent_query import router as agent_query_router
from app.api.v1.endpoints.browse import router as browse_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.options import router as options_router
from app.api.v1.endpoints.query import router as query_router

api_v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

api_v1_router.include_router(health_router)
api_v1_router.include_router(options_router)
api_v1_router.include_router(query_router)
api_v1_router.include_router(agent_query_router)
api_v1_router.include_router(browse_router)