"""
Document chunking utilities with content-defined chunking and deduplication.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    from fastcdc import fastcdc
    FASTCDC_AVAILABLE = True
except ImportError:
    FASTCDC_AVAILABLE = False

from schemas.base import ChunkMetadata, EvidenceSpan
from cli.utils.logging import gola_logger

@dataclass
class ChunkConfig:
    """Chunking configuration."""
    min_size: int = 512
    max_size: int = 4096
    avg_size: int = 2048
    hash_alg: str = "sha256"
    normalize_whitespace: bool = True
    remove_headers: bool = True
    remove_footers: bool = True
    merge_short_paragraphs: bool = True
    min_paragraph_length: int = 50
    preserve_sentence_boundaries: bool = True
    preserve_paragraph_boundaries: bool = True

class DocumentChunker:
    """Document chunker with content-defined chunking."""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize document chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        
        if not FASTCDC_AVAILABLE:
            gola_logger.warning("FastCDC not available. Using fallback chunking.")
    
    def chunk_document(self, text: str, file_id: str, 
                      char_offset: int = 0) -> List[ChunkMetadata]:
        """
        Chunk document text into content-defined chunks.
        
        Args:
            text: Document text
            file_id: Source file ID
            char_offset: Character offset from start of document
            
        Returns:
            List of chunk metadata
        """
        if not text.strip():
            return []
        
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Use FastCDC if available, otherwise fallback
        if FASTCDC_AVAILABLE:
            chunks = self._chunk_with_fastcdc(normalized_text, file_id, char_offset)
        else:
            chunks = self._chunk_with_fallback(normalized_text, file_id, char_offset)
        
        # Post-process chunks
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self._post_process_chunk(chunk)
            if processed_chunk:
                processed_chunks.append(processed_chunk)
        
        gola_logger.info(f"Created {len(processed_chunks)} chunks from document")
        return processed_chunks
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for chunking.
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text
        """
        if self.config.normalize_whitespace:
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        if self.config.remove_headers:
            # Remove common headers (simple heuristic)
            lines = text.split('\n')
            filtered_lines = []
            for line in lines:
                if not self._is_header_line(line):
                    filtered_lines.append(line)
            text = '\n'.join(filtered_lines)
        
        if self.config.remove_footers:
            # Remove common footers (simple heuristic)
            lines = text.split('\n')
            filtered_lines = []
            for line in lines:
                if not self._is_footer_line(line):
                    filtered_lines.append(line)
            text = '\n'.join(filtered_lines)
        
        return text
    
    def _is_header_line(self, line: str) -> bool:
        """Check if line is a header."""
        line = line.strip()
        
        # Page numbers
        if re.match(r'^\d+$', line):
            return True
        
        # Common header patterns
        header_patterns = [
            r'^page\s+\d+$',
            r'^\d+\s*of\s*\d+$',
            r'^confidential$',
            r'^draft$',
            r'^internal\s+use\s+only$'
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _is_footer_line(self, line: str) -> bool:
        """Check if line is a footer."""
        line = line.strip()
        
        # Common footer patterns
        footer_patterns = [
            r'^page\s+\d+$',
            r'^\d+\s*of\s*\d+$',
            r'^confidential$',
            r'^draft$',
            r'^internal\s+use\s+only$',
            r'^©\s*\d{4}',
            r'^all\s+rights\s+reserved'
        ]
        
        for pattern in footer_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _chunk_with_fastcdc(self, text: str, file_id: str, 
                           char_offset: int) -> List[ChunkMetadata]:
        """
        Chunk text using FastCDC.
        
        Args:
            text: Normalized text
            file_id: Source file ID
            char_offset: Character offset
            
        Returns:
            List of chunk metadata
        """
        chunks = []
        
        # Convert text to bytes for FastCDC
        text_bytes = text.encode('utf-8')
        
        # Run FastCDC
        cdc_chunks = fastcdc(
            text_bytes,
            min_size=self.config.min_size,
            max_size=self.config.max_size,
            avg_size=self.config.avg_size
        )
        
        # Convert CDC chunks to metadata
        for i, cdc_chunk in enumerate(cdc_chunks):
            chunk_text = cdc_chunk.data.decode('utf-8')
            
            # Calculate character positions
            start_pos = char_offset + cdc_chunk.offset
            end_pos = start_pos + len(chunk_text)
            
            # Generate chunk hash
            chunk_hash = self._generate_chunk_hash(chunk_text)
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_hash=chunk_hash,
                file_id=file_id,
                char_start=start_pos,
                char_end=end_pos,
                text_norm=chunk_text
            )
            
            chunks.append(chunk_metadata)
        
        return chunks
    
    def _chunk_with_fallback(self, text: str, file_id: str, 
                            char_offset: int) -> List[ChunkMetadata]:
        """
        Fallback chunking method when FastCDC is not available.
        
        Args:
            text: Normalized text
            file_id: Source file ID
            char_offset: Character offset
            
        Returns:
            List of chunk metadata
        """
        chunks = []
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_start = char_offset
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > self.config.max_size:
                # Finalize current chunk
                if current_chunk:
                    chunk_metadata = self._create_chunk_metadata(
                        current_chunk, file_id, current_start, 
                        char_offset + len(current_chunk)
                    )
                    chunks.append(chunk_metadata)
                
                # Start new chunk
                current_chunk = paragraph
                current_start = char_offset + len(current_chunk) - len(paragraph)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunk_metadata = self._create_chunk_metadata(
                current_chunk, file_id, current_start,
                char_offset + len(current_chunk)
            )
            chunks.append(chunk_metadata)
        
        return chunks
    
    def _create_chunk_metadata(self, text: str, file_id: str, 
                              start_pos: int, end_pos: int) -> ChunkMetadata:
        """
        Create chunk metadata.
        
        Args:
            text: Chunk text
            file_id: Source file ID
            start_pos: Start character position
            end_pos: End character position
            
        Returns:
            Chunk metadata
        """
        chunk_hash = self._generate_chunk_hash(text)
        
        return ChunkMetadata(
            chunk_hash=chunk_hash,
            file_id=file_id,
            char_start=start_pos,
            char_end=end_pos,
            text_norm=text
        )
    
    def _generate_chunk_hash(self, text: str) -> str:
        """
        Generate hash for chunk text.
        
        Args:
            text: Chunk text
            
        Returns:
            Chunk hash
        """
        if self.config.hash_alg == "sha256":
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        elif self.config.hash_alg == "md5":
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        else:
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _post_process_chunk(self, chunk: ChunkMetadata) -> Optional[ChunkMetadata]:
        """
        Post-process chunk.
        
        Args:
            chunk: Chunk metadata
            
        Returns:
            Processed chunk metadata or None if chunk should be discarded
        """
        # Skip very short chunks
        if len(chunk.text_norm.strip()) < self.config.min_paragraph_length:
            return None
        
        # Merge short paragraphs if enabled
        if self.config.merge_short_paragraphs:
            chunk.text_norm = self._merge_short_paragraphs(chunk.text_norm)
        
        # Preserve sentence boundaries if enabled
        if self.config.preserve_sentence_boundaries:
            chunk.text_norm = self._preserve_sentence_boundaries(chunk.text_norm)
        
        return chunk
    
    def _merge_short_paragraphs(self, text: str) -> str:
        """
        Merge short paragraphs.
        
        Args:
            text: Chunk text
            
        Returns:
            Text with merged paragraphs
        """
        paragraphs = text.split('\n\n')
        merged_paragraphs = []
        current_paragraph = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If current paragraph is short, merge with next
            if len(current_paragraph) < self.config.min_paragraph_length:
                if current_paragraph:
                    current_paragraph += " " + paragraph
                else:
                    current_paragraph = paragraph
            else:
                # Finalize current paragraph
                if current_paragraph:
                    merged_paragraphs.append(current_paragraph)
                current_paragraph = paragraph
        
        # Add final paragraph
        if current_paragraph:
            merged_paragraphs.append(current_paragraph)
        
        return '\n\n'.join(merged_paragraphs)
    
    def _preserve_sentence_boundaries(self, text: str) -> str:
        """
        Ensure chunk ends at sentence boundaries.
        
        Args:
            text: Chunk text
            
        Returns:
            Text with preserved sentence boundaries
        """
        # Simple sentence boundary detection
        sentence_end_pattern = r'[.!?]\s+'
        sentences = re.split(sentence_end_pattern, text)
        
        if len(sentences) <= 1:
            return text
        
        # If chunk is too long, truncate at sentence boundary
        if len(text) > self.config.max_size:
            truncated_text = ""
            for sentence in sentences:
                if len(truncated_text + sentence) <= self.config.max_size:
                    truncated_text += sentence + ". "
                else:
                    break
            
            return truncated_text.strip()
        
        return text
    
    def chunk_parsed_document(self, parsed_data: Dict[str, Any], 
                             file_id: str) -> List[ChunkMetadata]:
        """
        Chunk a parsed document.
        
        Args:
            parsed_data: Parsed document data
            file_id: Source file ID
            
        Returns:
            List of chunk metadata
        """
        blocks = parsed_data.get("blocks", [])
        
        # Extract text from blocks
        full_text = ""
        block_spans = []
        
        for block in blocks:
            start_pos = len(full_text)
            block_text = block.get("text", "") + "\n"
            full_text += block_text
            end_pos = len(full_text)
            
            block_spans.append({
                "start": start_pos,
                "end": end_pos,
                "type": block.get("type", "text"),
                "page_number": block.get("page_number")
            })
        
        # Chunk the full text
        chunks = self.chunk_document(full_text, file_id)
        
        # Add block information to chunks
        for chunk in chunks:
            chunk.metadata = {
                "block_types": self._get_chunk_block_types(chunk, block_spans),
                "page_numbers": self._get_chunk_page_numbers(chunk, block_spans)
            }
        
        return chunks
    
    def _get_chunk_block_types(self, chunk: ChunkMetadata, 
                              block_spans: List[Dict[str, Any]]) -> List[str]:
        """
        Get block types that overlap with chunk.
        
        Args:
            chunk: Chunk metadata
            block_spans: Block span information
            
        Returns:
            List of block types
        """
        block_types = []
        
        for block_span in block_spans:
            # Check if block overlaps with chunk
            if (block_span["start"] < chunk.char_end and 
                block_span["end"] > chunk.char_start):
                block_types.append(block_span["type"])
        
        return list(set(block_types))
    
    def _get_chunk_page_numbers(self, chunk: ChunkMetadata, 
                               block_spans: List[Dict[str, Any]]) -> List[int]:
        """
        Get page numbers that overlap with chunk.
        
        Args:
            chunk: Chunk metadata
            block_spans: Block span information
            
        Returns:
            List of page numbers
        """
        page_numbers = []
        
        for block_span in block_spans:
            # Check if block overlaps with chunk
            if (block_span["start"] < chunk.char_end and 
                block_span["end"] > chunk.char_start):
                page_number = block_span.get("page_number")
                if page_number:
                    page_numbers.append(page_number)
        
        return list(set(page_numbers))

# Global document chunker instance
document_chunker = DocumentChunker()

def get_document_chunker() -> DocumentChunker:
    """Get the global document chunker instance."""
    return document_chunker

def init_document_chunker(config: Optional[ChunkConfig] = None) -> DocumentChunker:
    """
    Initialize the global document chunker.
    
    Args:
        config: Chunking configuration
        
    Returns:
        Document chunker instance
    """
    global document_chunker
    document_chunker = DocumentChunker(config)
    return document_chunker 