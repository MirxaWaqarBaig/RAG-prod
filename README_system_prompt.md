# System Prompt RAG (Prompt-Only)

A trimmed-down version of the production Graph RAG server focused on contextual chunking, Chroma vector retrieval, and a unified system-prompt generation path. Neo4j knowledge-graph features and SQL/business-data fusion are deliberately removed to keep the stack lightweight.

## Contents
- `core.py`, `core_hybrid.py`, `cache.py`: Retrieval helpers, hybrid fusion, optional PostgreSQL cache manager.
- `optimized_chunking.py`, `local_qwen_llm.py`: Contextual chunk pipeline and optional Qwen enhancer.
- `build_chroma_only.py`: CLI to rebuild contextual chunks and refresh the Chroma index.
- `system_rag_service.py`, `system_rag_server.py`: System-prompt RAG service and ZMQ entrypoint.
- `rag_system_prompt.md`: System prompt used for both concise and detailed responses.
- `input/`, `input_chunked/`, `chromadb/`, `cache/`: Document sources, generated chunks, vector store, and auxiliary cache data.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Populate `input/` with `.txt` documents.
3. Run `python build_chroma_only.py` to generate contextual chunks and build the Chroma index.
4. Launch the server: `python system_rag_server.py serve`

Environment variables can be stored in `.env.example` (copy to `.env`), e.g. LLM credentials. PostgreSQL cache is optional—if the connection fails the server simply operates without caching.
