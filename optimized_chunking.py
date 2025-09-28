#!/usr/bin/env python3
"""
Optimized Chunking Strategy for Graph RAG
Implements dynamic chunking with size parameters and overlap
"""

import re
from typing import List
import core

class OptimizedChunking:
    """Dynamic chunking strategy with size optimization and overlap"""
    
    def __init__(self):
        self.chunk_size_min = 800
        self.chunk_size_max = 1200
        self.overlap_size = 250
    
    def detect_document_type(self, content: str) -> str:
        """Dynamically detect document type based on content patterns"""
        
        # Count Q&A patterns
        qa_patterns = len(re.findall(r'\n\d+\.\d+\s*Q：', content))
        qa_ratio = qa_patterns / max(len(content.split('\n')), 1)
        
        # Count paragraph breaks
        paragraph_breaks = len(re.findall(r'\n\n+', content))
        paragraph_ratio = paragraph_breaks / max(len(content.split('\n')), 1)
        
        # Count headers (markdown-style)
        headers = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        header_ratio = headers / max(len(content.split('\n')), 1)
        
        # Determine document type
        if qa_ratio > 0.1:  # More than 10% Q&A patterns
            return "qa_document"
        elif header_ratio > 0.05:  # More than 5% headers
            return "structured_document"
        elif paragraph_ratio > 0.2:  # More than 20% paragraph breaks
            return "paragraph_document"
        else:
            return "mixed_document"
    
    def qa_chunking_with_size_limit(self, content: str) -> List[str]:
        """Chunk Q&A documents with size limits"""
        qa_pairs = re.split(r'\n\d+\.\d+\s*Q：', content)
        chunks = []
        
        for qa in qa_pairs:
            qa = qa.strip()
            if not qa:
                continue
                
            if len(qa) <= self.chunk_size_max:
                chunks.append(qa)
            else:
                # Split large Q&A while preserving Q&A structure
                sub_chunks = self.split_qa_with_size_limit(qa)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def split_qa_with_size_limit(self, qa_text: str) -> List[str]:
        """Split large Q&A while preserving structure"""
        if len(qa_text) <= self.chunk_size_max:
            return [qa_text]
        
        # Try to split at sentence boundaries
        sentences = re.split(r'[。！？]', qa_text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.chunk_size_max:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def structured_chunking_with_size_limit(self, content: str) -> List[str]:
        """Chunk structured documents with size limits"""
        # Split by headers
        sections = re.split(r'^#+\s+', content, flags=re.MULTILINE)
        chunks = []
        
        for section in sections:
            if not section.strip():
                continue
                
            if len(section) <= self.chunk_size_max:
                chunks.append(section.strip())
            else:
                # Split large sections
                sub_chunks = self.split_section_with_size_limit(section)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def split_section_with_size_limit(self, section: str) -> List[str]:
        """Split large sections while preserving structure"""
        if len(section) <= self.chunk_size_max:
            return [section.strip()]
        
        # Split by subsections
        subsections = re.split(r'^\d+\.\d+\s+', section, flags=re.MULTILINE)
        chunks = []
        current_chunk = ""
        
        for subsection in subsections:
            if not subsection.strip():
                continue
                
            if len(current_chunk + subsection) <= self.chunk_size_max:
                current_chunk += subsection + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = subsection + "\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def paragraph_chunking_with_size_limit(self, content: str) -> List[str]:
        """Chunk by paragraphs with size limits"""
        paragraphs = re.split(r'\n\n+', content)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            if len(current_chunk + paragraph) <= self.chunk_size_max:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def hybrid_chunking_with_size_limit(self, content: str) -> List[str]:
        """Hybrid approach with size limits"""
        chunks = []
        
        # First try Q&A chunking
        qa_chunks = self.qa_chunking_with_size_limit(content)
        
        for chunk in qa_chunks:
            if 'Q：' in chunk:
                chunks.append(chunk)
            else:
                # Apply paragraph chunking to non-Q&A content
                para_chunks = self.paragraph_chunking_with_size_limit(chunk)
                chunks.extend(para_chunks)
        
        return chunks
    
    def add_chunk_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks to preserve context"""
        chunks_with_overlap = []
        
        for i, chunk in enumerate(chunks):
            if i > 0 and len(chunks[i-1]) > self.overlap_size:
                # Add end of previous chunk as overlap
                prev_end = chunks[i-1][-self.overlap_size:]
                enhanced_chunk = f"上下文：{prev_end}\n\n{chunk}"
            else:
                enhanced_chunk = chunk
                
            chunks_with_overlap.append(enhanced_chunk)
        
        return chunks_with_overlap
    
    def optimize_to_target_size(self, chunks: List[str]) -> List[str]:
        """Optimize chunks to target size range"""
        optimized = []
        
        for chunk in chunks:
            if self.chunk_size_min <= len(chunk) <= self.chunk_size_max:
                optimized.append(chunk)
            elif len(chunk) < self.chunk_size_min:
                # Try to merge with next chunk if possible
                if optimized and len(optimized[-1] + chunk) <= self.chunk_size_max:
                    optimized[-1] += "\n\n" + chunk
                else:
                    optimized.append(chunk)
            else:
                # Split large chunks
                sub_chunks = self.split_to_target_size(chunk)
                optimized.extend(sub_chunks)
        
        return optimized
    
    def split_to_target_size(self, chunk: str) -> List[str]:
        """Split large chunks to target size"""
        if len(chunk) <= self.chunk_size_max:
            return [chunk]
        
        # Try to split at sentence boundaries
        sentences = re.split(r'[。！？]', chunk)
        sub_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.chunk_size_max:
                current_chunk += sentence + "。"
            else:
                if current_chunk and len(current_chunk) >= self.chunk_size_min:
                    sub_chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk and len(current_chunk) >= self.chunk_size_min:
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks
    
    def process_document_dynamically(self, content: str, filename: str = None) -> List[str]:
        """Complete dynamic document processing pipeline with optimized parameters"""
        
        print(f"🔍 Analyzing document type...")
        doc_type = self.detect_document_type(content)
        print(f"📄 Detected document type: {doc_type}")
        
        # Apply appropriate chunking with size limits
        if doc_type == "qa_document":
            chunks = self.qa_chunking_with_size_limit(content)
        elif doc_type == "structured_document":
            chunks = self.structured_chunking_with_size_limit(content)
        elif doc_type == "paragraph_document":
            chunks = self.paragraph_chunking_with_size_limit(content)
        else:
            chunks = self.hybrid_chunking_with_size_limit(content)
        
        print(f"✂️ Created {len(chunks)} initial chunks")
        
        # Add overlap (200-300 characters)
        chunks_with_overlap = self.add_chunk_overlap(chunks)
        print(f"🔄 Added overlap between chunks")
        
        # Optimize to target size range (800-1200)
        optimized_chunks = self.optimize_to_target_size(chunks_with_overlap)
        
        print(f"⚡ Optimized to {len(optimized_chunks)} chunks")
        if optimized_chunks:
            avg_size = sum(len(c) for c in optimized_chunks) // len(optimized_chunks)
            print(f"📏 Average chunk size: {avg_size} characters")
        
        return optimized_chunks
    
    def enhance_chunks_with_context(self, chunks: List[str], filename: str, doc_type: str = None, llm=None) -> List[str]:
        """Add simple AI-generated context to existing chunks"""
        print(f"🚀 Adding contextual enhancement to {len(chunks)} chunks from {filename}")
        
        # Use provided LLM instance or load one if not provided
        if llm is None:
            try:
                from local_qwen_llm import LocalQwenLLM
                llm = LocalQwenLLM()
                print(f"  🤖 Qwen model loaded successfully")
            except Exception as e:
                print(f"  ❌ Failed to load Qwen model: {e}")
                print(f"  📝 Falling back to original chunks without enhancement")
                return chunks
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Simple prompt for context generation
                prompt = f"""请为以下文本内容提供一个简洁的上下文摘要（30-80字）：

{chunk[:800]}

请直接输出摘要，不要其他内容。"""

                # Get context from Qwen
                context = llm.complete(prompt)
                context = str(context).strip()
                
                # Clean up the context
                context = context.replace("这是", "").replace("该", "").replace("这个", "").strip()
                
                # Simple validation - just check if we got something reasonable
                if len(context) > 5 and len(context) < 200:
                    enhanced_chunk = f"上下文：{context}\n\n{chunk}"
                    enhanced_chunks.append(enhanced_chunk)
                    print(f"  ✅ Enhanced chunk {i+1}/{len(chunks)}")
                else:
                    print(f"  ⚠️ Context too short/long for chunk {i+1}, using original")
                    enhanced_chunks.append(chunk)
                    
            except Exception as e:
                print(f"  ⚠️ Failed to enhance chunk {i+1}: {e}")
                # Fallback to original chunk if enhancement fails
                enhanced_chunks.append(chunk)
        
        print(f"🎉 Contextual enhancement completed for {filename}")
        return enhanced_chunks
    
    def enhance_single_chunk_with_llm(self, chunk: str, filename: str, doc_type: str, llm) -> str:
        """Legacy method - now handled by enhance_chunks_with_context"""
        # This method is kept for backward compatibility but not used in the new approach
        return chunk
    
    def enhance_single_chunk(self, chunk: str, filename: str, doc_type: str) -> str:
        """Legacy method for backward compatibility"""
        try:
            from local_qwen_llm import LocalQwenLLM
            llm = LocalQwenLLM()
            return self.enhance_single_chunk_with_llm(chunk, filename, doc_type, llm)
        except Exception as e:
            print(f"Error enhancing chunk: {e}")
            return chunk
