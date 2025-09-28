#!/usr/bin/env python3
"""Rebuild contextual chunks and the Chroma index (no Neo4j/SQL dependencies)."""

import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure we can import sibling modules when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core  # type: ignore
import optimized_chunking  # type: ignore

try:
    import local_qwen_llm  # type: ignore
except ImportError:  # Optional dependency
    local_qwen_llm = None


INPUT_DIR = PROJECT_ROOT / "input"
CHUNKED_DIR = PROJECT_ROOT / "input_chunked"
CHROMA_DIR = PROJECT_ROOT / "chromadb"
CACHE_DIR = PROJECT_ROOT / "cache"

TRACKING_TIMESTAMPS = CACHE_DIR / "document_timestamps.json"
TRACKING_HASHES = CACHE_DIR / "document_hashes.json"


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def list_txt_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.txt") if p.is_file()])


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"Warning: failed to write {path}: {exc}")


def _file_hash(path: Path) -> str:
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def check_document_changes() -> Tuple[List[Path], List[Path], List[Path]]:
    timestamps = _load_json(TRACKING_TIMESTAMPS)
    hashes = _load_json(TRACKING_HASHES)

    new_docs: List[Path] = []
    modified_docs: List[Path] = []
    unchanged_docs: List[Path] = []

    for doc in list_txt_files(INPUT_DIR):
        doc_mtime = doc.stat().st_mtime
        doc_hash = _file_hash(doc)
        stamp_key = doc.name

        recorded_hash = hashes.get(stamp_key)
        if stamp_key not in timestamps:
            print(f"📄 New document detected: {doc.name}")
            new_docs.append(doc)
        elif recorded_hash != doc_hash:
            print(f"📄 Modified document detected: {doc.name}")
            modified_docs.append(doc)
        else:
            unchanged_docs.append(doc)

        timestamps[stamp_key] = doc_mtime
        hashes[stamp_key] = doc_hash

    # Remove records for deleted files
    existing = {doc.name for doc in list_txt_files(INPUT_DIR)}
    deleted = set(list(_load_json(TRACKING_TIMESTAMPS).keys())) - existing
    for name in deleted:
        timestamps.pop(name, None)
        hashes.pop(name, None)

    _save_json(TRACKING_TIMESTAMPS, timestamps)
    _save_json(TRACKING_HASHES, hashes)

    return new_docs, modified_docs, unchanged_docs


def _reset_tracking_files() -> None:
    for path in (TRACKING_TIMESTAMPS, TRACKING_HASHES):
        if path.exists():
            try:
                path.unlink()
                print(f"🗑️  Removed {path}")
            except Exception as exc:
                print(f"Warning: could not remove {path}: {exc}")


def _reset_chroma_dir() -> None:
    if CHROMA_DIR.exists():
        try:
            shutil.rmtree(CHROMA_DIR)
            print(f"🗑️  Removed existing Chroma directory: {CHROMA_DIR}")
        except Exception as exc:
            print(f"Warning: could not remove {CHROMA_DIR}: {exc}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def build_chromadb_with_optimized_chunking(process_only: List[Path] | None, use_context_enhancement: bool) -> None:
    chunker = optimized_chunking.OptimizedChunking()

    # Optionally load local Qwen model for contextual summaries
    llm = None
    if use_context_enhancement and local_qwen_llm is not None:
        try:
            print("🤖 Loading Local Qwen LLM for contextual summaries...")
            llm = local_qwen_llm.LocalQwenLLM()
            print("✅ Qwen model loaded.")
        except Exception as exc:
            print(f"❌ Failed to load Qwen model: {exc}. Falling back to standard chunks.")
            llm = None
            use_context_enhancement = False

    files_to_process = process_only if process_only is not None else list_txt_files(INPUT_DIR)
    if not files_to_process:
        print("✅ No documents to process.")
        return

    # Clean chunk directory for deterministic build when processing all docs
    if process_only is None:
        for existing in CHUNKED_DIR.glob("*"):
            existing.unlink(missing_ok=True)

    total_chunks = 0
    for doc_path in files_to_process:
        content = doc_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunker.process_document_dynamically(content, filename=doc_path.name)

        if use_context_enhancement and llm:
            print(f"🚀 Enhancing chunks for {doc_path.name}...")
            chunks = chunker.enhance_chunks_with_context(chunks, doc_path.name, llm=llm)

        for idx, chunk in enumerate(chunks, start=1):
            chunk_file = CHUNKED_DIR / f"{doc_path.stem}_chunk_{idx:03d}.txt"
            chunk_file.write_text(chunk, encoding="utf-8")

        print(f"✂️  Wrote {len(chunks)} chunks for {doc_path.name}")
        total_chunks += len(chunks)

    if total_chunks == 0:
        print("⚠️ No chunks generated; skipping Chroma rebuild.")
        return

    core.load_embedding_model(CACHE_DIR)
    core.create_index(CHROMA_DIR, CHUNKED_DIR)
    print("✅ Chroma index refreshed.")


def main() -> None:
    ensure_dirs()

    if not any(INPUT_DIR.iterdir()):
        print("❌ No documents found in data/input. Add .txt files and rerun.")
        sys.exit(1)

    new_docs, modified_docs, unchanged_docs = check_document_changes()

    print("\n=== Chroma Build Mode ===")
    print("  1) Standard chunking")
    print("  2) Enhanced chunking with AI context")
    enhancement_choice = input("Choose enhancement [1]: ").strip() or "1"
    use_context_enhancement = enhancement_choice == "2"
    if use_context_enhancement:
        print("🚀 Contextual enhancement enabled.")
    else:
        print("📝 Standard chunking selected.")

    has_changes = bool(new_docs or modified_docs)
    print("\nDetected document status:")
    print(f"  New: {len(new_docs)} | Modified: {len(modified_docs)} | Unchanged: {len(unchanged_docs)}")

    print("\n=== Build Scope ===")
    if has_changes:
        print("  1) Process changed files only (incremental)")
        print("  2) Rebuild all documents")
        print("  0) Exit")
    else:
        print("  1) Rebuild all documents")
        print("  2) Force rebuild all documents")
        print("  0) Exit")

    scope_choice = input("Enter choice [1]: ").strip() or "1"

    if scope_choice == "0":
        print("Exiting without changes.")
        return

    if has_changes and scope_choice == "1":
        docs_to_process = new_docs + modified_docs
        if not docs_to_process:
            print("✅ No documents require processing.")
            return
        print(f"📋 Processing {len(docs_to_process)} file(s): {[p.name for p in docs_to_process]}")
    else:
        docs_to_process = None
        _reset_tracking_files()
        _reset_chroma_dir()
        print("🔥 Rebuilding Chroma from all documents.")

    build_chromadb_with_optimized_chunking(docs_to_process, use_context_enhancement)

    print("\n🎉 Chroma build completed successfully!")


if __name__ == "__main__":
    main()
