from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question")

class Citation(BaseModel):
    source: str = Field(..., description="Source document or URL")
    page: Optional[int] = Field(None, description="1-based page number when available")
    snippet: str = Field(..., description="Short evidence excerpt from the source")
    score: Optional[float] = Field(None, description="Retriever relevance score from 0 to 1")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The AI generated answer")
    topic: str = Field(..., description="Classified topic")
    sources: List[str] = Field(..., description="List of source documents used")
    confidence: float = Field(0.0, description="Answer confidence from 0 to 1")
    citations: List[Citation] = Field(default_factory=list, description="Evidence snippets used")
    refined_question: str = Field("", description="Standalone question used for retrieval")
    needs_more_context: bool = Field(False, description="Whether the answer needs more source material")

class UrlIngestRequest(BaseModel):
    url: str = Field(..., description="URL to scrape and ingest")

class DeleteSourceRequest(BaseModel):
    source_path: str = Field(..., description="The exact source path to delete")

class IngestResponse(BaseModel):
    status: str
    source: str
    chunks_ingested: int = 0
    skipped_reason: Optional[str] = None
