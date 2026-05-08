"""
FastAPI application entry point.
Provides async REST endpoints for complaint submission, status, and audit retrieval.
"""
from __future__ import annotations

import time
import warnings

import structlog

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.api.routes.complaints import router as complaints_router
from src.api.routes.audit import router as audit_router
from src.config.settings import get_settings
from src.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY
from src.schemas.models import HealthResponse

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="Systematic Compliance Agent API",
        description=(
            "Multi-agent AI pipeline for autonomous classification, routing, investigation, "
            "and resolution of consumer financial complaints at scale."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    app.include_router(complaints_router, prefix="/api/v1/complaints", tags=["Complaints"])
    app.include_router(audit_router, prefix="/api/v1/audit", tags=["Audit"])

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        return response

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        return HealthResponse()

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
