"""Hybrid retrieval helpers (semantic + BM25) without KG/SQL dependencies."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from llama_index.core import Settings
from llama_index.core.schema import TextNode

import core

from core import (  # noqa: F401 - re-exported helpers used by callers
    cache_model,
    create_index,
    fallback_llm,
    load_embedding_model,
    load_index,
    set_llm,
)


class BM25Retriever:
    """Simple BM25 keyword retriever with mixed Chinese/English support."""

    def __init__(self, documents: List[Dict[str, Any]], k1: float = 1.2, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)

        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: List[int] = []
        self.avg_doc_length = 0.0

        self._build_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        tokens: List[str] = []

        for match in re.finditer(r"\b[\w]+\b", text):
            word = match.group(0).lower()
            if len(word) >= 2:
                tokens.append(word)

        for seq_match in re.finditer(r"[\u4e00-\u9fff]+", text):
            seq = seq_match.group(0)
            tokens.extend(list(seq))
            if len(seq) >= 2:
                tokens.extend([seq[i : i + 2] for i in range(len(seq) - 1)])

        return tokens

    def _build_index(self) -> None:
        total_length = 0

        for doc_id, doc in enumerate(self.documents):
            tokens = self._tokenize(doc["text"].lower())
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            total_length += doc_length

            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for term, freq in term_freq.items():
                self.inverted_index[term].append((doc_id, freq))

        self.avg_doc_length = total_length / self.doc_count if self.doc_count else 0.0

    def _idf(self, term: str) -> float:
        doc_freq = len(self.inverted_index.get(term, ()))
        if doc_freq == 0:
            return 0.0
        return math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def _score(self, term: str, doc_id: int, term_freq: int) -> float:
        idf = self._idf(term)
        doc_length = self.doc_lengths[doc_id]
        return idf * (term_freq * (self.k1 + 1)) / (
            term_freq + self.k1 * (1 - self.b + self.b * doc_length / (self.avg_doc_length or 1.0))
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        query_tokens = self._tokenize(query.lower())
        doc_scores: Dict[int, float] = defaultdict(float)

        for term in query_tokens:
            for doc_id, term_freq in self.inverted_index.get(term, ()):  # type: ignore[arg-type]
                doc_scores[doc_id] += self._score(term, doc_id, term_freq)

        ranked = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return [(self.documents[doc_id], score) for doc_id, score in ranked[:top_k]]


class HybridRetriever:
    """Blend semantic retrieval with BM25 keyword scoring."""

    def __init__(self, index, embedding_model, alpha: float = 0.6, bm25_documents: List[Dict[str, Any]] | None = None) -> None:
        self.semantic_retriever = index.as_retriever(similarity_top_k=10)
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.bm25_retriever: BM25Retriever | None = None
        self._build_bm25_index(bm25_documents)

    def _build_bm25_index(self, bm25_documents: List[Dict[str, Any]] | None) -> None:
        try:
            if bm25_documents:
                self.bm25_retriever = BM25Retriever(bm25_documents)
                print(f"[Hybrid] Built BM25 index with corpus: {len(bm25_documents)} documents")
            else:
                print("[Hybrid] No corpus provided for BM25; using semantic only")
        except Exception as exc:
            print(f"[Hybrid] Failed to build BM25 index: {exc}")
            self.bm25_retriever = None

    def rebuild_bm25(self, bm25_documents: List[Dict[str, Any]] | None) -> None:
        if not bm25_documents:
            print("[Hybrid] Rebuild skipped: empty corpus")
            return
        try:
            self.bm25_retriever = BM25Retriever(bm25_documents)
            print(f"[Hybrid] Rebuilt BM25 index with {len(bm25_documents)} documents")
        except Exception as exc:
            print(f"[Hybrid] Failed to rebuild BM25 index: {exc}")

    def retrieve(self, query: str, top_k: int = 5):
        semantic_nodes = self.semantic_retriever.retrieve(query)
        bm25_results: List[Tuple[Dict[str, Any], float]] = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query, top_k=10)
        return self._fuse_results(semantic_nodes, bm25_results, top_k)

    def _fuse_results(self, semantic_nodes, bm25_results, top_k: int):
        semantic_scores: Dict[str, Dict[str, Any]] = {}
        for idx, node in enumerate(semantic_nodes):
            semantic_scores[node.text] = {"score": 1.0 / (idx + 1), "node": node}

        bm25_scores: Dict[str, Dict[str, Any]] = {}
        for idx, (doc_data, score) in enumerate(bm25_results):
            bm25_scores[doc_data["text"]] = {
                "score": 1.0 / (idx + 1),
                "doc_data": doc_data,
                "bm25_score": score,
            }

        combined: Dict[str, Dict[str, Any]] = {}
        for text in set(semantic_scores) | set(bm25_scores):
            semantic_score = semantic_scores.get(text, {}).get("score", 0.0)
            keyword_score = bm25_scores.get(text, {}).get("score", 0.0)
            combined_score = self.alpha * semantic_score + (1 - self.alpha) * keyword_score

            if text in semantic_scores:
                node = semantic_scores[text]["node"]
                if hasattr(node, "metadata") and text in bm25_scores:
                    node.metadata["bm25_score"] = bm25_scores[text]["bm25_score"]
            else:
                doc_data = bm25_scores[text]["doc_data"]
                node = TextNode(text=doc_data["text"], metadata=doc_data.get("metadata", {}))
                node.metadata["bm25_score"] = bm25_scores[text]["bm25_score"]

            combined[text] = {"score": combined_score, "node": node}

        ranked = sorted(combined.values(), key=lambda entry: entry["score"], reverse=True)
        return [entry["node"] for entry in ranked[:top_k]]


def calculate_cosine_similarity(embedding1, embedding2) -> float:
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if denom == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / denom)


def run_query(prompt: str, index, cache_manager=None, embedding_model=None):
    embedding_model = embedding_model or Settings.embed_model
    query_embedding = embedding_model.get_text_embedding(prompt)

    if cache_manager:
        cached = cache_manager.get_cached_response(prompt, query_embedding, embedding_model)
        if cached:
            print("[Hybrid RAG] Cache hit")
            return cached["response"]

    print("[Hybrid RAG] Cache miss, performing hybrid retrieval…")
    retriever = HybridRetriever(index, embedding_model)
    nodes = retriever.retrieve(prompt, top_k=10)

    if not nodes:
        print("[Hybrid RAG] No results, falling back to LLM")
        return fallback_llm(prompt, cache_manager)

    sources: List[str] = []
    sections: List[str] = []
    for idx, node in enumerate(nodes, start=1):
        source_file = node.metadata.get("file_name", "Unknown Source")
        sources.append(source_file)
        semantic_score = getattr(node, "score", 0.0)
        bm25_score = node.metadata.get("bm25_score", 0.0)
        if semantic_score == 0.0 and bm25_score == 0.0:
            try:
                node_embedding = embedding_model.get_text_embedding(node.text)
                semantic_score = calculate_cosine_similarity(query_embedding, node_embedding)
            except Exception:
                semantic_score = 0.8
        score_info = f"Semantic: {semantic_score:.2f}"
        if bm25_score:
            score_info += f", BM25: {bm25_score:.2f}"
        sections.append(f"Document {idx} ({score_info}) [{source_file}]\n{node.text.strip()}")

    enhanced_prompt = f"""You are a helpful assistant. Use the following context retrieved with semantic and keyword search to answer the query.

QUERY: {prompt}

CONTEXT:
{chr(10).join(sections)}

Answer clearly and directly. Avoid referring to the documents explicitly.
"""

    try:
        response = Settings.llm.complete(enhanced_prompt)
        if cache_manager:
            cache_manager.add_entry(prompt, str(response), sources)
        return str(response)
    except Exception as exc:
        print(f"[Hybrid RAG] LLM completion failed: {exc}")
        return fallback_llm(prompt, cache_manager)
