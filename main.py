import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import DeleteSourceRequest, IngestResponse, QueryRequest, QueryResponse, UrlIngestRequest
from rag_engine import RAGEngine
from security import is_admin_authorized, safe_upload_filename, validate_public_url


load_dotenv()


def _allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost:8501"]


def create_app(engine=None, admin_api_key: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Gallery AI RAG API", description="Commercial RAG API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rag_engine = engine or RAGEngine()
    upload_dir = os.getenv("UPLOAD_DIRECTORY", "./data/uploads")
    max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    configured_admin_key = admin_api_key if admin_api_key is not None else os.getenv("ADMIN_API_KEY")
    os.makedirs(upload_dir, exist_ok=True)

    def require_admin(x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key")):
        if not is_admin_authorized(x_admin_key, configured_admin_key):
            raise HTTPException(status_code=401, detail="Invalid or missing admin key")

    @app.get("/")
    def health_check():
        return {"status": "online", "service": "Gallery AI"}

    @app.post("/query", response_model=QueryResponse)
    def ask_ai(request: QueryRequest):
        result = rag_engine.query(request.question)
        if isinstance(result, QueryResponse):
            return result
        answer, topic, sources = result
        return QueryResponse(answer=answer, topic=topic, sources=sources)

    @app.post("/query_stream")
    def ask_ai_stream(request: QueryRequest):
        return StreamingResponse(rag_engine.stream_query(request.question), media_type="text/plain")

    @app.post("/admin/ingest-pdf", response_model=IngestResponse, dependencies=[Depends(require_admin)])
    async def ingest_pdf(file: UploadFile = File(...)):
        filename = safe_upload_filename(file.filename)
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        content = await file.read()
        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds MAX_UPLOAD_BYTES")

        file_location = os.path.join(upload_dir, filename)
        with open(file_location, "wb") as file_object:
            file_object.write(content)

        try:
            count = rag_engine.ingest_pdf(file_location)
            return IngestResponse(status="success", source=filename, chunks_ingested=count)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/ingest-url", response_model=IngestResponse, dependencies=[Depends(require_admin)])
    def ingest_url(request: UrlIngestRequest):
        try:
            public_url = validate_public_url(request.url)
            count = rag_engine.ingest_url(public_url)
            return IngestResponse(status="success", source=public_url, chunks_ingested=count)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/admin/sources", dependencies=[Depends(require_admin)])
    def get_sources():
        return {"sources": rag_engine.list_sources()}

    @app.delete("/admin/source", dependencies=[Depends(require_admin)])
    def delete_source(request: DeleteSourceRequest):
        success, msg = rag_engine.delete_source(request.source_path)
        if not success:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "success", "message": msg}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
