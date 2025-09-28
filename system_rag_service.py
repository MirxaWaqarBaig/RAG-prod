"""System Prompt RAG service: contextual retrieval with concise/detailed responses."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cache as cache_module  # type: ignore
import core  # type: ignore
import core_hybrid  # type: ignore


class SystemRagService:
    """Contextual chunk + Chroma RAG service without knowledge graph or SQL."""

    cache_dir = PROJECT_ROOT / ".cache"
    input_dir = PROJECT_ROOT / "input"
    chunk_dir = PROJECT_ROOT / "input_chunked"
    db_dir = PROJECT_ROOT / "chromadb"
    system_prompt_path = PROJECT_ROOT / "rag_system_prompt.md"

    def __init__(self) -> None:
        self._ensure_directories()

        # System prompt mode only
        self.system_mode = "system_prompt"
        print("[SystemRag] Running in system_prompt mode")

        self.system_prompt = self._load_system_prompt()
        self.latency_tracker: Dict[str, float] = {}
        self.temp_context_store: Optional[Dict[str, str]] = None
        self.last_contexts: Dict[str, object] = {}

        # Optional PostgreSQL-backed cache
        self.cache_manager: Optional[cache_module.SemanticCacheManager]
        try:
            self.cache_manager = cache_module.SemanticCacheManager()
            print("[SystemRag] Semantic cache initialized")
        except Exception as exc:  # pragma: no cover - cache optional
            self.cache_manager = None
            print(f"[SystemRag] Cache disabled ({exc})")

        # Load embedding model and index
        self.embedding_model = core.load_embedding_model(self.cache_dir)
        if not any(self.db_dir.glob("**/*")):
            raise FileNotFoundError(
                "ChromaDB index not found. Run build_chroma_only.py before starting the service."
            )
        self.index = core.load_index(self.db_dir)

        # Build hybrid retriever (semantic + BM25)
        bm25_corpus = self._load_bm25_corpus_from_chroma()
        self.hybrid_retriever = core_hybrid.HybridRetriever(
            index=self.index,
            embedding_model=self.embedding_model,
            alpha=0.6,
            bm25_documents=bm25_corpus,
        )

        # Track last change in source documents
        self.last_index_time = self._get_latest_input_mtime()
        self.reload_interval = int(os.environ.get("RAG_RELOAD_INTERVAL", "300"))

        # Configure LLM
        core.set_llm()
        print("[SystemRag] LLM initialized")

        # Background reloader thread
        self._reloader = threading.Thread(target=self._index_auto_reloader, daemon=True)
        self._reloader.start()
        print(f"[SystemRag] Auto-reload interval: {self.reload_interval} seconds")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _ensure_directories(self) -> None:
        for path in [self.cache_dir, self.input_dir, self.chunk_dir, self.db_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _load_system_prompt(self) -> str:
        if not self.system_prompt_path.exists():
            return ""
        return self.system_prompt_path.read_text(encoding="utf-8").strip()

    def _get_latest_input_mtime(self) -> float:
        mtimes = [p.stat().st_mtime for p in self.input_dir.glob("*.txt")]
        return max(mtimes, default=0.0)

    def _load_bm25_corpus_from_chroma(self) -> List[Dict[str, object]]:
        try:
            client = chromadb.PersistentClient(path=str(self.db_dir))
            collection = client.get_collection("document_collection")
        except Exception as exc:
            print(f"[SystemRag] Failed to access Chroma collection: {exc}")
            return []

        corpus: List[Dict[str, object]] = []
        offset = 0
        page_size = 1000
        while True:
            batch = collection.get(
                include=["documents", "metadatas"],  # ids are returned by default in newer Chroma versions
                offset=offset,
                limit=page_size,
            )
            ids = batch.get("ids", []) or []
            docs = batch.get("documents", []) or []
            metas = batch.get("metadatas", []) or []
            if not ids:
                break
            for idx, doc_id in enumerate(ids):
                text = docs[idx] if idx < len(docs) else None
                meta = metas[idx] if idx < len(metas) else {}
                if text:
                    corpus.append({"text": text, "metadata": meta or {}, "node_id": doc_id})
            offset += len(ids)

        print(f"[SystemRag] BM25 corpus loaded: {len(corpus)} documents")
        return corpus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_admin_command(self, command: str) -> str:
        try:
            if command == "!reload":
                self.index = core.load_index(self.db_dir)
                bm25_corpus = self._load_bm25_corpus_from_chroma()
                self.hybrid_retriever = core_hybrid.HybridRetriever(
                    index=self.index,
                    embedding_model=self.embedding_model,
                    alpha=0.6,
                    bm25_documents=bm25_corpus,
                )
                self.last_index_time = self._get_latest_input_mtime()
                return "[Admin] Index and BM25 corpus reloaded"

            if command == "!clear":
                if not self.cache_manager:
                    return "[Admin] Cache is disabled"
                result = self.cache_manager.clear_cache()
                return f"[Admin] {result['message']}"

            if command == "!stats":
                stats = ["[Admin] System RAG stats:"]
                stats.append(f"  Mode: {self.system_mode}")
                stats.append(f"  Reload interval: {self.reload_interval}s")
                stats.append(f"  Last input change: {datetime.fromtimestamp(self.last_index_time)}")
                stats.append(f"  Cache enabled: {'yes' if self.cache_manager else 'no'}")
                stats.append(f"  Detailed context ready: {'yes' if self.temp_context_store else 'no'}")
                return "\n".join(stats)

            if command == "!detailed":
                detailed = self._handle_detailed_request()
                return detailed.get("response", "No detailed response available")

            return f"[Admin] Unknown command: {command}"

        except Exception as exc:
            traceback.print_exc()
            return f"[Admin] Error: {exc}"

    def handle_query(self, prompt: str) -> str:
        if prompt.startswith("!"):
            return self.handle_admin_command(prompt)

        try:
            result = self.process_query(prompt)
            return result.get("response", "No response generated")
        except Exception as exc:
            traceback.print_exc()
            return f"[SystemRag Error] {exc}"

    def process_query(self, query: str) -> Dict[str, object]:
        start_time = time.time()
        self.latency_tracker = {}

        # Query encoding
        encode_start = time.time()
        query_embedding = self.embedding_model.get_text_embedding(query)
        self.latency_tracker["query_encoding"] = time.time() - encode_start

        # Cache lookup
        cache_hit = False
        cached = None
        cache_start = time.time()
        if self.cache_manager:
            cached = self.cache_manager.get_cached_response(
                query,
                query_embedding,
                self.embedding_model,
                variant="normal",
                skip_frequency_increment=False,
            )
        self.latency_tracker["cache_check"] = time.time() - cache_start

        context_bundle = self._build_context_bundle(query)
        integrated_context = context_bundle["integrated_context"]
        self._store_context_for_detailed(query, integrated_context)
        self.last_contexts = {"document": context_bundle}

        if cached:
            cache_hit = True
            response_text = cached["response"]
        else:
            response_text = self._generate_response_system_prompt(query, context_bundle, mode="NORMAL")

            if self.cache_manager:
                self.cache_manager.add_entry(query, response_text, context_bundle["sources"], variant="normal")

        # Previously we appended a manual detailed-answer prompt here, but the UI now
        # provides its own button so we return the raw response text instead.
        response_with_option = response_text

        total_time = time.time() - start_time
        self.latency_tracker["total"] = total_time

        return {
            "response": response_with_option,
            "cache_hit": cache_hit,
            "response_time": total_time,
            "method": "system_prompt",
            "latency_breakdown": self.latency_tracker,
        }

    # ------------------------------------------------------------------
    # Context construction and response helpers
    # ------------------------------------------------------------------
    def _build_context_bundle(self, query: str, top_k: int = 6) -> Dict[str, object]:
        retrieve_start = time.time()
        nodes = self.hybrid_retriever.retrieve(query, top_k=top_k)
        self.latency_tracker["document_retrieval"] = time.time() - retrieve_start

        snippets: List[str] = []
        sources: List[str] = []
        for idx, node in enumerate(nodes, start=1):
            source = node.metadata.get("file_name", node.metadata.get("source", "Unknown Source"))
            sources.append(source)
            snippet = node.text.strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + "…"
            snippets.append(f"[{idx}] {source}\n{snippet}")

        if not snippets:
            snippets.append("No relevant document context retrieved.")

        integrated = "\n\n".join(snippets)
        return {
            "snippets": snippets,
            "sources": sources,
            "integrated_context": f"Query: {query}\n\nDOCUMENT CONTEXT:\n{integrated}",
        }

    def _generate_response_system_prompt(self, query: str, context_bundle: Dict[str, object], mode: str) -> str:
        integrated_context = context_bundle["integrated_context"]
        prompt = f"""{self.system_prompt}

## QUERY
{query}

## CONTEXT
{integrated_context}

## MODE
{mode}

Generate a {mode} response following the guidelines above."""
        try:
            response = core.Settings.llm.complete(prompt)
            return str(response)
        except Exception as exc:
            print(f"[SystemRag] System prompt response failed: {exc}")
            return core.fallback_llm(query, self.cache_manager)

    def _store_context_for_detailed(self, query: str, context: str) -> None:
        self.temp_context_store = {
            "query": query,
            "context": context,
            "timestamp": time.time(),
        }

    def _handle_detailed_request(self) -> Dict[str, object]:
        if not self.temp_context_store:
            return {
                "response": "No previous query context available. Please ask a question first.",
                "cache_hit": False,
            }

        if time.time() - self.temp_context_store["timestamp"] > 1800:
            self.temp_context_store = None
            return {
                "response": "Previous query context has expired. Please ask your question again.",
                "cache_hit": False,
            }

        original_query = self.temp_context_store["query"]
        context = self.temp_context_store["context"]

        if self.cache_manager:
            query_embedding = self.embedding_model.get_text_embedding(original_query)
            cached = self.cache_manager.get_cached_response(
                original_query,
                query_embedding,
                self.embedding_model,
                variant="detailed",
            )
            if cached:
                return {
                    "response": cached["response"],
                    "cache_hit": True,
                }

        context_bundle = self.last_contexts.get("document", {"integrated_context": context})
        detailed_response = self._generate_response_system_prompt(original_query, context_bundle, mode="DETAILED")

        if self.cache_manager:
            self.cache_manager.add_entry(
                original_query,
                detailed_response,
                ["System RAG Detailed"],
                variant="detailed",
            )

        return {
            "response": detailed_response,
            "cache_hit": False,
        }

    # ------------------------------------------------------------------
    # Background reload loop
    # ------------------------------------------------------------------
    def _index_auto_reloader(self) -> None:
        print("[SystemRag] Monitoring input/ for changes…")
        while True:
            time.sleep(self.reload_interval)
            try:
                latest_mtime = self._get_latest_input_mtime()
                if latest_mtime > self.last_index_time:
                    print("[SystemRag] Changes detected. Reloading index…")
                    self.index = core.load_index(self.db_dir)
                    bm25_corpus = self._load_bm25_corpus_from_chroma()
                    self.hybrid_retriever = core_hybrid.HybridRetriever(
                        index=self.index,
                        embedding_model=self.embedding_model,
                        alpha=0.6,
                        bm25_documents=bm25_corpus,
                    )
                    self.last_index_time = latest_mtime
                    print("[SystemRag] Reload complete")
            except Exception as exc:
                print(f"[SystemRag] Reload failed: {exc}")
