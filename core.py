import os
import torch
import numpy as np
from pathlib import Path
from typing import List

from transformers import BertTokenizer, BertModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from langchain_openai.chat_models.base import BaseChatOpenAI
from llama_index.llms.langchain import LangChainLLM


def set_llm(llm_type: str = None):
    """Configure and set the LLM to be used (Qwen or DeepSeek)."""
    llm_type = llm_type or os.environ.get("LLM_TYPE", "deepseek")

    if llm_type == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        llm = BaseChatOpenAI(
            model="qwen-max",
            openai_api_key=api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_tokens=1024
        )
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        llm = BaseChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1024
        )

    Settings.llm = LangChainLLM(llm=llm)


def load_embedding_model(cache_dir: Path):
    """Load a local cached HuggingFace embedding model."""
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Embedding] Using device: {device}")

    model = HuggingFaceEmbedding(
        model_name=str(cache_dir),
        device=device,
        normalize=True
    )
    Settings.embed_model = model
    return model


def cache_model(cache_dir: Path):
    """Download and cache the tokenizer and model to local directory."""
    model_name = os.environ.get("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    tokenizer.save_pretrained(str(cache_dir))
    model.save_pretrained(str(cache_dir))
    print(f"[Embedding] Model cached to {cache_dir}")


def create_index(db_dir: Path, input_dir: Path):
    """Create a vector index of documents in input_dir and store it in db_dir."""

    # Ensure required directories exist
    db_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    # Check if input_dir has any files
    if not any(input_dir.iterdir()):
        raise ValueError(f"No documents found in input_dir: {input_dir}")

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(db_dir))
    collection = chroma_client.get_or_create_collection("document_collection")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load documents
    documents = SimpleDirectoryReader(str(input_dir)).load_data()
    print(f"[Index] Loaded {len(documents)} documents from {input_dir}")

    # Build the index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context
    )
    print(f"[Index] Index created and stored in {db_dir}")
    return index

def load_index(db_dir: Path):
    """Load an existing index from a ChromaDB path."""
    if not db_dir.exists():
        raise ValueError(f"Invalid db_dir: {db_dir}")

    chroma_client = chromadb.PersistentClient(path=str(db_dir))

    try:
        collection = chroma_client.get_collection("document_collection")
    except ValueError:
        raise ValueError("No existing index found. Run create-index first.")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    return index


def run_query(prompt: str, index, cache_manager=None, embedding_model=None):
    """Perform a query through the RAG pipeline: cache → retrieval → LLM."""
    if not embedding_model:
        embedding_model = Settings.embed_model

    query_embedding = embedding_model.get_text_embedding(prompt)

    if cache_manager:
        cached = cache_manager.get_cached_response(prompt, query_embedding, embedding_model)
        if cached:
            print("[RAG] Cache hit")
            return cached["response"]

    # Enhanced Retrieval and Query Processing
    print("[RAG] Cache miss, retrieving documents...")
    
    # Use retriever directly for more control
    top_k = int(os.environ.get('RAG_TOP_K', 10))  # Configurable, default 10
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(prompt)
    
    if not retrieved_nodes:
        print("[RAG] No relevant documents found, using fallback LLM")
        return fallback_llm(prompt, cache_manager)
    
    # Format retrieved context with enhanced template
    context_sections = []
    source_info = []
    
    for i, node in enumerate(retrieved_nodes, 1):
        # Extract metadata
        source_file = node.metadata.get("file_name", "Unknown Source")
        source_info.append(source_file)
        
        # Calculate relevance score (similarity)
        relevance_score = getattr(node, 'score', 0.0)
        if relevance_score == 0.0:
            # Fallback: calculate similarity if not available
            try:
                node_embedding = embedding_model.get_text_embedding(node.text)
                query_emb = embedding_model.get_text_embedding(prompt)
                relevance_score = calculate_cosine_similarity(query_emb, node_embedding)
            except:
                relevance_score = 0.8  # Default score
        
        # Format context section
        context_sections.append(f"Document {i} (Relevance: {relevance_score:.2f}): [{source_file}]\n{node.text.strip()}")
    
    # Create enhanced prompt template
    enhanced_prompt = f"""You are a knowledgeable assistant specializing in business operations and services. 
Based on the following relevant documents, provide a comprehensive and accurate answer.

QUERY: {prompt}

RELEVANT CONTEXT:
{chr(10).join(context_sections)}

INSTRUCTIONS:
1. Answer directly and concisely based on the provided context
2. Use specific details and examples from the context when relevant
3. If the context doesn't fully address the query, clearly state what information is available and what is missing
4. Maintain a professional and helpful tone
5. For Chinese queries, respond in Chinese; for English queries, respond in English

ANSWER:"""

    try:
        # Call LLM directly with enhanced prompt
        llm = Settings.llm
        response = llm.complete(enhanced_prompt)
        
        # Cache the response
        if cache_manager:
            cache_manager.add_entry(prompt, str(response), source_info)
        
        return str(response)

    except Exception as e:
        print(f"[RAG] Enhanced query processing failed: {e}")
        return fallback_llm(prompt, cache_manager)


def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    # Convert to numpy arrays
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def fallback_llm(prompt: str, cache_manager=None):
    """Fallback completion using only the LLM (no retrieval)."""
    llm = Settings.llm
    system_prompt = f"""You are an AI assistant helping with technical questions.
I don't have specific context for your query, but I'll try to provide a helpful response.

Question: {prompt}

Answer:"""

    try:
        response = llm.complete(system_prompt)

        if cache_manager:
            cache_manager.add_entry(prompt, str(response), [])

        return str(response)

    except Exception as err:
        return f"[LLM Fallback Error] {err}"
