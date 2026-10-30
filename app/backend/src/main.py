import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.config import settings
from src.chat_database import init_chat_database
from src.database import ensure_collection

logger = logging.getLogger("second_brain")

app = FastAPI(
    title="Second Brain AI API",
    description="Fahad's Second Brain - RAG-powered AI assistant with personalized communication style",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*", "X-Visitor-Id"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    origin = request.headers.get("origin")
    headers = {}
    if origin:
        normalized_origin = origin.rstrip("/")
        allowed = {o.rstrip("/") for o in settings.CORS_ORIGINS}
        if normalized_origin in allowed or "*" in settings.CORS_ORIGINS:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            headers["Access-Control-Allow-Methods"] = "*"
            headers["Access-Control-Allow-Headers"] = "*"
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
        headers=headers,
    )

@app.on_event("startup")
def initialize_database():
    try:
        init_chat_database()
    except Exception as e:
        print(f"Could not initialize database on startup: {e}")

@app.get("/health/live", response_model=dict)
def liveness_check():
    return {"status": "ok"}


@app.get("/health/ready", response_model=dict)
def readiness_check():
    qdrant_ready = ensure_collection(settings.EMBEDDING_DIMENSION)
    return {
        "status": "ok" if qdrant_ready else "degraded",
        "qdrant": "connected" if qdrant_ready else "error",
        "gemini_configured": bool(settings.GEMINI_API_KEYS),
    }


@app.get("/health", response_model=dict)
def health_check():
    return readiness_check()

from src.routes import router
app.include_router(router)
from src.admin_routes import router as admin_router
app.include_router(admin_router)

@app.get("/")
async def root():
    return {
        "message": "Second Brain AI API",
        "version": "0.1.0",
        "docs": "/docs",
    }

if __name__ == "__main__":
    import uvicorn  # only needed for local `python src/main.py` runs
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
