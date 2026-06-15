import os
import re
from typing import List, Generator, Tuple
from dotenv import load_dotenv

# Load .env before LangChain loaders inspect environment variables.
load_dotenv()

# LangChain Core Components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from models import Citation, QueryResponse

# --- 1. Environment and Basic Configuration ---
MIN_RETRIEVAL_CONFIDENCE = float(os.getenv("MIN_RETRIEVAL_CONFIDENCE", "0.45"))


def get_persist_directory() -> str:
    return os.getenv("PERSIST_DIRECTORY", "./data/chroma_db")


def _clean_snippet(text: str, limit: int) -> str:
    snippet = re.sub(r"\s+", " ", text or "").strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: max(0, limit - 1)].rstrip() + "…"


def _normalize_score(score) -> float:
    if score is None:
        return 0.0
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, value)), 3)


def build_citations(doc_scores, snippet_chars: int = 260) -> List[Citation]:
    citations = []
    seen = set()
    for doc, score in doc_scores:
        source = doc.metadata.get("source", "Unknown")
        raw_page = doc.metadata.get("page")
        page = raw_page + 1 if isinstance(raw_page, int) else raw_page
        snippet = _clean_snippet(doc.page_content, snippet_chars)
        key = (source, page, snippet)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source=source,
                page=page,
                snippet=snippet,
                score=_normalize_score(score),
            )
        )
    return citations


def calculate_confidence(doc_scores) -> float:
    scores = [_normalize_score(score) for _, score in doc_scores]
    return max(scores) if scores else 0.0


def needs_more_context(sources: List[str], confidence: float, threshold: float = MIN_RETRIEVAL_CONFIDENCE) -> bool:
    return not sources or confidence < threshold


def detect_response_language(question: str) -> str:
    return "Chinese" if re.search(r"[\u4e00-\u9fff]", question or "") else "English"


def response_language_instruction(question: str) -> str:
    language = detect_response_language(question)
    if language == "Chinese":
        return "Respond in Chinese because the user asked in Chinese."
    return "Respond in English because the user asked in English."


# --- 2. RAG Core Engine Class ---
class RAGEngine:
    def __init__(self):
        # Startup log in English
        print("🛠️ [SYSTEM] Starting Engine: Memory + MMR Rerank + Source Tracking enabled")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            persist_directory=get_persist_directory(),
            embedding_function=self.embeddings
        )

        # MMR (Maximal Marginal Relevance) reduces redundant chunks and improves diversity
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6}
        )

        # Planner: used for query rewriting and topic classification (non-streaming)
        self.planner = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # Main LLM: used for final answer generation
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

        # Persistent conversation memory
        self.memory = ChatMessageHistory()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # ------------------------------------------------------------------
    # Query Rewriting
    # ------------------------------------------------------------------
    def _rewrite_query(self, question: str) -> str:
        """Rewrite ambiguous questions using recent conversation history."""
        if not self.memory.messages:
            return question

        history_snippet = "\n".join(
            [f"{m.type}: {m.content}" for m in self.memory.messages[-4:]]
        )
        # Prompt translated to English for logic consistency
        prompt = (
            f"Based on the following conversation history, rewrite the user's new question into a standalone and complete search term.\n\n"
            f"Conversation History:\n{history_snippet}\n\n"
            f"New Question: {question}\n\n"
            f"Output only the rewritten text without any explanation."
        )
        refined = self.planner.invoke(prompt).content.strip()
        print(f"🔍 [LOG] Intent Rewriting Result: {refined}")
        return refined

    # ------------------------------------------------------------------
    # Topic Classification
    # ------------------------------------------------------------------
    def _classify_topic(self, question: str, context: str) -> str:
        """Classify the topic of the answer based on question and retrieved context."""
        prompt = (
            f"Based on the question and references, summarize the topic in 2–4 English words (e.g., Product Info, Tech Support, Pricing, History).\n\n"
            f"Question: {question}\n"
            f"Context Summary: {context[:500]}\n\n"
            f"Output only the topic category words."
        )
        topic = self.planner.invoke(prompt).content.strip()
        return topic if topic else "General"

    # ------------------------------------------------------------------
    # Source Extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_sources(docs) -> List[str]:
        """Extract unique source filenames/URLs from retrieved documents."""
        seen = set()
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    @staticmethod
    def _extract_sources_from_citations(citations: List[Citation]) -> List[str]:
        seen = set()
        sources = []
        for citation in citations:
            if citation.source not in seen:
                seen.add(citation.source)
                sources.append(citation.source)
        return sources

    def _retrieve_with_scores(self, query: str):
        """Retrieve documents with relevance scores, falling back to MMR docs if needed."""
        try:
            return self.vector_store.similarity_search_with_relevance_scores(query, k=5)
        except Exception as e:
            print(f"⚠️ [WARN] Scored retrieval failed, falling back to MMR: {e}")
            docs = self.retriever.invoke(query)
            return [(doc, 0.5) for doc in docs]

    # ------------------------------------------------------------------
    # Main Query Interface (sync, used by FastAPI /query)
    # ------------------------------------------------------------------
    def query(self, question: str) -> QueryResponse:
        """Synchronous query: returns a commercial-grade structured response."""
        print(f"\n🚀 [LOG] Received request: {question}")

        # 1. Query rewriting
        refined_q = self._rewrite_query(question)

        # 2. Local vector retrieval with evidence scores
        doc_scores = self._retrieve_with_scores(refined_q)
        docs = [doc for doc, _ in doc_scores]
        print(f"📚 [LOG] Retrieval complete, chunks recalled: {len(docs)}")

        citations = build_citations(doc_scores)
        sources = self._extract_sources_from_citations(citations)
        confidence = calculate_confidence(doc_scores)

        # 3. If nothing trustworthy is retrieved, respond honestly
        if needs_more_context(sources, confidence):
            answer = "I'm sorry, I couldn't find any information related to this question in the knowledge base. Please upload relevant PDFs or URLs first."
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)
            return QueryResponse(
                answer=answer,
                topic="No relevant data",
                sources=sources,
                confidence=confidence,
                citations=citations,
                refined_question=refined_q,
                needs_more_context=True,
            )

        local_context = "\n\n".join([d.page_content for d in docs])

        # 4. Confidence check
        check_prompt = (
            f"You are a strict knowledge base assistant. Here is the retrieved context:\n{local_context}\n\n"
            f"User Question: {question}\n\n"
            f"Does the context provide enough information to answer the question? Answer only YES or NO."
        )
        decision = self.planner.invoke(check_prompt).content.strip().upper()

        if "NO" in decision:
            answer = (
                "I'm sorry, I cannot accurately answer this question based on the current knowledge base. "
                "Please try uploading more relevant materials."
            )
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)
            return QueryResponse(
                answer=answer,
                topic="Insufficient Coverage",
                sources=sources,
                confidence=confidence,
                citations=citations,
                refined_question=refined_q,
                needs_more_context=True,
            )

        # 5. Build prompt and generate answer
        language_instruction = response_language_instruction(question)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional knowledge base assistant. "
                "Strictly use the provided [Context] to answer the question. "
                "If the information is not in the context, say 'Information not mentioned'. "
                "Be concise, accurate, and cite only facts supported by the context. "
                f"{language_instruction}"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "[Context]:\n{context}\n\n[User Question]: {question}")
        ])

        formatted_messages = prompt.format_messages(
            context=local_context,
            question=question,
            history=self.memory.messages
        )

        print("✍️ [LOG] Generating answer...")
        answer_chunks = []
        for chunk in self.llm.stream(formatted_messages):
            answer_chunks.append(chunk.content)
        answer = "".join(answer_chunks)

        # 6. Classify topic
        topic = self._classify_topic(question, local_context)

        # 7. Update memory
        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)
        print(f"💾 [LOG] Memory synced. Session complete. Topic: {topic}\n")

        return QueryResponse(
            answer=answer,
            topic=topic,
            sources=sources,
            confidence=confidence,
            citations=citations,
            refined_question=refined_q,
            needs_more_context=False,
        )

    # ------------------------------------------------------------------
    # Streaming Query Interface (used by /query_stream)
    # ------------------------------------------------------------------
    def stream_query(self, question: str) -> Generator[str, None, None]:
        """Streaming query: yields answer tokens one by one."""
        print(f"\n🚀 [LOG] Received streaming request: {question}")

        refined_q = self._rewrite_query(question)
        docs = self.retriever.invoke(refined_q)
        print(f"📚 [LOG] Retrieval complete, chunks recalled: {len(docs)}")

        if not docs:
            msg = "I'm sorry, I couldn't find any information related to this question in the knowledge base."
            self.memory.add_user_message(question)
            self.memory.add_ai_message(msg)
            yield msg
            return

        local_context = "\n\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional knowledge base assistant. "
                "Strictly use the provided [Context] to answer. "
                "If not found, say 'Information not mentioned'. Use English."
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "[Context]:\n{context}\n\n[User Question]: {question}")
        ])

        formatted_messages = prompt.format_messages(
            context=local_context,
            question=question,
            history=self.memory.messages
        )

        accumulated_answer = ""
        for chunk in self.llm.stream(formatted_messages):
            token = chunk.content
            accumulated_answer += token
            yield token

        self.memory.add_user_message(question)
        self.memory.add_ai_message(accumulated_answer)
        print("💾 [LOG] Streaming memory synced.\n")

    # ------------------------------------------------------------------
    # Knowledge Base Management
    # ------------------------------------------------------------------
    def ingest_pdf(self, path: str) -> int:
        """Load a PDF, split into chunks, and add to the vector store."""
        print(f"📄 [SYSTEM] Parsing PDF: {path}")
        loader = PyPDFLoader(path)
        documents = loader.load_and_split(self.splitter)
        # Normalize source metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(path)
        self.vector_store.add_documents(documents)
        print(f"✅ [SYSTEM] PDF Ingestion complete. {len(documents)} chunks added.")
        return len(documents)

    def ingest_url(self, url: str) -> int:
        """Scrape a URL, split into chunks, and add to the vector store."""
        print(f"🔗 [SYSTEM] Scraping URL: {url}")
        loader = WebBaseLoader(url)
        documents = loader.load_and_split(self.splitter)
        for doc in documents:
            doc.metadata["source"] = url
        self.vector_store.add_documents(documents)
        print(f"✅ [SYSTEM] URL Ingestion complete. {len(documents)} chunks added.")
        return len(documents)

    def list_sources(self) -> List[str]:
        """Return a deduplicated list of all ingested sources."""
        try:
            all_docs = self.vector_store.get()
            metadatas = all_docs.get("metadatas", [])
            sources = list({m.get("source", "Unknown") for m in metadatas if m})
            return sorted(sources)
        except Exception as e:
            print(f"❌ [ERROR] Failed to list sources: {e}")
            return []

    def delete_source(self, source_path: str) -> Tuple[bool, str]:
        """Delete all chunks belonging to a specific source from the vector store."""
        try:
            all_docs = self.vector_store.get()
            ids_to_delete = [
                doc_id
                for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"])
                if meta.get("source") == source_path
            ]
            if not ids_to_delete:
                return False, f"Source not found: {source_path}"
            self.vector_store.delete(ids=ids_to_delete)
            print(f"🗑️ [SYSTEM] Deleted source '{source_path}', removed {len(ids_to_delete)} chunks.")
            return True, f"Successfully deleted {len(ids_to_delete)} chunks (Source: {source_path})"
        except Exception as e:
            print(f"❌ [ERROR] Deletion failed: {e}")
            return False, str(e)
