from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from .config import Settings, get_settings
from .models import (
    DocumentInfo,
    FeedbackRecord,
    FeedbackRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from .rag_pipeline import RAGPipeline

# Logging 
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

# Application lifecycle

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    settings = get_settings()
    logger.info("startup", app=settings.app_name, version=settings.app_version)
    _pipeline = RAGPipeline(settings)
    yield
    logger.info("shutdown")


# FastAPI app

def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description="Internal company knowledge assistant with RAG",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics at /metrics
    Instrumentator().instrument(app).expose(app)

    # Serve frontend static files if directory exists
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/ui", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

    # Request timing middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        response.headers["X-Response-Time-Ms"] = str(int((time.time() - start) * 1000))
        return response

    # Exception handlers 
    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception):
        logger.error("unhandled_error", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # Routes 
    register_routes(app, cfg)
    return app


# Helper: extract auth headers 

def _parse_user(
    x_user_id: str = Header(default="anonymous"),
    x_user_groups: str = Header(default=""),
) -> tuple[str, list[str]]:
    groups = [g.strip() for g in x_user_groups.split(",") if g.strip()]
    return x_user_id, groups


def _get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return _pipeline


# Route Registration
def register_routes(app: FastAPI, cfg: Settings) -> None:

    # Health
    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    async def health(pipeline: RAGPipeline = Depends(_get_pipeline)):
        try:
            chunk_count = pipeline.vector_store.count()
            qdrant_status = "healthy"
        except Exception:
            chunk_count = -1
            qdrant_status = "unhealthy"

        overall = "healthy" if qdrant_status == "healthy" else "degraded"
        return HealthResponse(
            status=overall,
            version=cfg.app_version,
            components={
                "qdrant": qdrant_status,
                "bm25_index": "healthy" if pipeline.bm25_index._bm25 else "not_built",
                "chunk_count": str(chunk_count),
            },
        )

    # Query (non-streaming)
    @app.post("/api/v1/query", response_model=QueryResponse, tags=["rag"])
    async def query(
        req: QueryRequest,
        pipeline: RAGPipeline = Depends(_get_pipeline),
        auth: tuple = Depends(_parse_user),
    ):
        user_id, user_groups = auth
        # Header-supplied identity overrides body if present
        effective_user = req.user_id or user_id
        effective_groups = req.user_groups or user_groups

        logger.info("query_received", user=effective_user, query=req.query[:80])

        response = await pipeline.query(
            query=req.query,
            user_id=effective_user,
            user_groups=effective_groups,
            top_k=req.top_k,
        )
        return response

    # Query (streaming SSE)
    @app.post("/api/v1/query/stream", tags=["rag"])
    async def query_stream(
        req: QueryRequest,
        pipeline: RAGPipeline = Depends(_get_pipeline),
        auth: tuple = Depends(_parse_user),
    ):
        user_id, user_groups = auth
        effective_user = req.user_id or user_id
        effective_groups = req.user_groups or user_groups

        async def event_generator():
            async for chunk in pipeline.stream_query(
                query=req.query,
                user_id=effective_user,
                user_groups=effective_groups,
                top_k=req.top_k,
            ):
                yield chunk

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Feedback
    @app.post("/api/v1/feedback", tags=["feedback"])
    async def feedback(
        req: FeedbackRequest,
        pipeline: RAGPipeline = Depends(_get_pipeline),
    ):
        record = FeedbackRecord(
            query_id=req.query_id,
            user_id=req.user_id,
            rating=req.rating,
            comment=req.comment,
        )
        fb_path = Path(cfg.feedback_path)
        fb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fb_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        return {"status": "recorded", "feedback_id": record.feedback_id}

    # Document listing 
    @app.get("/api/v1/documents", response_model=list[DocumentInfo], tags=["documents"])
    async def list_documents(pipeline: RAGPipeline = Depends(_get_pipeline)):
        payloads = pipeline.vector_store.get_all_chunks()
        seen: dict[str, DocumentInfo] = {}
        for p in payloads:
            did = p.get("document_id", "")
            meta = p.get("metadata", {})
            if did and did not in seen:
                seen[did] = DocumentInfo(
                    document_id=did,
                    title=meta.get("title", "Unknown"),
                    source_path=meta.get("source_path", ""),
                    department=meta.get("department"),
                    country=meta.get("country"),
                    document_type=meta.get("document_type", ""),
                    author=meta.get("author"),
                )
            elif did in seen:
                seen[did].chunk_count += 1
        return list(seen.values())

    # Ingest (trigger ingestion of a single document)
    @app.post("/api/v1/ingest", response_model=IngestResponse, tags=["ingestion"])
    async def ingest(req: IngestRequest):
        from ingestion.ingest_pdfs import IngestionPipeline

        cfg_local = get_settings()
        ingestor = IngestionPipeline(cfg_local)
        try:
            result = ingestor.ingest_file(
                source_path=req.source_path,
                document_type=req.document_type,
                title=req.title,
                department=req.department,
                country=req.country,
                author=req.author,
                acl_groups=req.acl_groups,
            )
            return result
        except Exception as e:
            logger.error("ingestion_failed", path=req.source_path, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Query log (internal / admin)
    @app.get("/api/v1/admin/query-log", tags=["admin"])
    async def query_log(limit: int = 100):
        log_path = Path(cfg.query_log_path)
        if not log_path.exists():
            return []
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-limit:]]


app = create_app()
