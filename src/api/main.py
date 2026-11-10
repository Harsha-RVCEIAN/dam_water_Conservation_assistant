"""FastAPI Application Entrypoint for KRS Dam Inflow Forecasting Service.

Implements lifespan management for one-time model loading, route configuration,
OpenAPI metadata, and exception handling.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.model_loader import ModelArtifactHolder, load_registered_serving_model
from src.api.routes.health import router as health_router
from src.api.routes.model_info import router as model_info_router
from src.api.routes.prediction import router as prediction_router

logger = logging.getLogger("krs_dam_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: load registered model once at startup."""
    logger.info("Initializing KRS Dam Inflow Serving Service...")
    try:
        # Load registered model Version 1 from MLflow
        model_holder = load_registered_serving_model()
        app.state.model_holder = model_holder
        logger.info(
            "Registered model '%s' (v%s) loaded successfully into application state.",
            model_holder.model_name,
            model_holder.version,
        )
    except Exception as exc:
        logger.error("Failed to load registered model during startup: %s", exc)
        # Store holder with error so /health still functions and /ready reports 503
        app.state.model_holder = ModelArtifactHolder(
            load_error=str(exc),
        )

    yield

    logger.info("Shutting down KRS Dam Inflow Serving Service.")


def create_app() -> FastAPI:
    """Application factory for FastAPI serving."""
    application = FastAPI(
        title="KRS Dam Inflow Forecasting API",
        description=(
            "Advisory machine learning inference service for K.R.Sagara Dam inflow forecasting. "
            "Exposes the registered Tuned Random Forest model (Version 1) over HTTP for "
            "water conservation planning and decision simulation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Register Routers
    application.include_router(health_router)
    application.include_router(model_info_router)
    application.include_router(prediction_router)

    # Custom Exception Handlers
    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Sanitize error inputs to ensure strictly valid JSON even if input contained NaN / Infinity
        import math
        clean_errors = []
        for err in exc.errors():
            err_dict = dict(err)
            inp = err_dict.get("input")
            if isinstance(inp, float):
                if math.isnan(inp):
                    err_dict["input"] = "NaN"
                elif math.isinf(inp):
                    err_dict["input"] = "Infinity" if inp > 0 else "-Infinity"
            clean_errors.append(err_dict)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": jsonable_encoder(clean_errors),
                "error_type": "ValidationError",
                "message": "Invalid request schema or feature values.",
            },
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred during request processing.",
                "error_type": type(exc).__name__,
            },
        )

    return application


app = create_app()
