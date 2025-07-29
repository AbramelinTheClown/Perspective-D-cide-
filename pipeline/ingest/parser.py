"""
Document parsing utilities using unstructured library.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

try:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import (
        Title, NarrativeText, ListItem, Table, Header, Footer,
        PageBreak, Address, Figure, Formula, Text
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from schemas.base import FileMetadata, EvidenceSpan
from cli.utils.logging import gola_logger

class DocumentBlock:
    """Represents a parsed document block."""
    
    def __init__(self, element, page_number: Optional[int] = None):
        """
        Initialize document block.
        
        Args:
            element: Unstructured element
            page_number: Page number (if available)
        """
        self.element = element
        self.page_number = page_number
        self.text = self._extract_text(element)
        self.block_type = self._get_block_type(element)
        self.metadata = self._extract_metadata(element)
    
    def _extract_text(self, element) -> str:
        """Extract text from element."""
        if hasattr(element, 'text'):
            return element.text
        elif hasattr(element, 'content'):
            return element.content
        else:
            return str(element)
    
    def _get_block_type(self, element) -> str:
        """Get block type from element."""
        if isinstance(element, Title):
            return "title"
        elif isinstance(element, NarrativeText):
            return "narrative"
        elif isinstance(element, ListItem):
            return "list_item"
        elif isinstance(element, Table):
            return "table"
        elif isinstance(element, Header):
            return "header"
        elif isinstance(element, Footer):
            return "footer"
        elif isinstance(element, PageBreak):
            return "page_break"
        elif isinstance(element, Address):
            return "address"
        elif isinstance(element, Figure):
            return "figure"
        elif isinstance(element, Formula):
            return "formula"
        else:
            return "text"
    
    def _extract_metadata(self, element) -> Dict[str, Any]:
        """Extract metadata from element."""
        metadata = {}
        
        if hasattr(element, 'metadata'):
            metadata.update(element.metadata)
        
        # Extract coordinates if available
        if hasattr(element, 'coordinates'):
            metadata['coordinates'] = element.coordinates
        
        # Extract bounding box if available
        if hasattr(element, 'bbox'):
            metadata['bbox'] = element.bbox
        
        return metadata

class DocumentParser:
    """Document parser using unstructured library."""
    
    def __init__(self, ocr_enabled: bool = True, ocr_language: str = 'eng'):
        """
        Initialize document parser.
        
        Args:
            ocr_enabled: Enable OCR for images
            ocr_language: OCR language
        """
        self.ocr_enabled = ocr_enabled and TESSERACT_AVAILABLE
        self.ocr_language = ocr_language
        
        if not UNSTRUCTURED_AVAILABLE:
            gola_logger.warning("Unstructured library not available. Document parsing disabled.")
        
        if ocr_enabled and not TESSERACT_AVAILABLE:
            gola_logger.warning("Tesseract not available. OCR disabled.")
            self.ocr_enabled = False
    
    def parse_document(self, file_path: Path, file_metadata: FileMetadata) -> Dict[str, Any]:
        """
        Parse document and extract structured content.
        
        Args:
            file_path: Path to the document
            file_metadata: File metadata
            
        Returns:
            Parsed document data
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise RuntimeError("Unstructured library not available")
        
        try:
            gola_logger.info(f"Parsing document: {file_path}")
            
            # Parse document
            elements = partition(
                filename=str(file_path),
                include_metadata=True,
                include_page_breaks=True
            )
            
            # Process elements
            blocks = []
            page_breaks = []
            current_page = 1
            
            for element in elements:
                if isinstance(element, PageBreak):
                    page_breaks.append(len(blocks))
                    current_page += 1
                else:
                    block = DocumentBlock(element, current_page)
                    blocks.append(block)
            
            # Extract document notes
            document_notes = self._extract_document_notes(blocks, file_metadata)
            
            # Create parsed document data
            parsed_data = {
                "file_id": file_metadata.file_id,
                "file_sha256": file_metadata.file_sha256,
                "parsed_at": datetime.utcnow().isoformat(),
                "total_blocks": len(blocks),
                "total_pages": current_page,
                "page_breaks": page_breaks,
                "blocks": [
                    {
                        "text": block.text,
                        "type": block.block_type,
                        "page_number": block.page_number,
                        "metadata": block.metadata
                    }
                    for block in blocks
                ],
                "document_notes": document_notes
            }
            
            gola_logger.info(f"Parsed {len(blocks)} blocks from {current_page} pages")
            return parsed_data
        
        except Exception as e:
            gola_logger.error(f"Error parsing document {file_path}: {e}")
            raise
    
    def _extract_document_notes(self, blocks: List[DocumentBlock], 
                               file_metadata: FileMetadata) -> Dict[str, Any]:
        """
        Extract document notes and layout information.
        
        Args:
            blocks: Parsed document blocks
            file_metadata: File metadata
            
        Returns:
            Document notes dictionary
        """
        notes = {
            "file_info": {
                "path": str(file_metadata.path),
                "size_bytes": file_metadata.size_bytes,
                "mime_type": file_metadata.mime_type,
                "language": file_metadata.language,
                "pii_level": file_metadata.pii_level
            },
            "layout": {
                "total_blocks": len(blocks),
                "block_types": {},
                "page_count": 0,
                "has_tables": False,
                "has_figures": False,
                "has_formulas": False
            },
            "content": {
                "total_text_length": 0,
                "title_count": 0,
                "list_item_count": 0,
                "table_count": 0,
                "figure_count": 0,
                "formula_count": 0
            },
            "ocr_stats": {
                "ocr_used": False,
                "ocr_confidence": None,
                "ocr_language": self.ocr_language
            }
        }
        
        # Analyze blocks
        page_numbers = set()
        total_text_length = 0
        
        for block in blocks:
            # Count block types
            block_type = block.block_type
            notes["layout"]["block_types"][block_type] = notes["layout"]["block_types"].get(block_type, 0) + 1
            
            # Count content types
            if block_type == "title":
                notes["content"]["title_count"] += 1
            elif block_type == "list_item":
                notes["content"]["list_item_count"] += 1
            elif block_type == "table":
                notes["content"]["table_count"] += 1
                notes["layout"]["has_tables"] = True
            elif block_type == "figure":
                notes["content"]["figure_count"] += 1
                notes["layout"]["has_figures"] = True
            elif block_type == "formula":
                notes["content"]["formula_count"] += 1
                notes["layout"]["has_formulas"] = True
            
            # Track page numbers
            if block.page_number:
                page_numbers.add(block.page_number)
            
            # Calculate text length
            total_text_length += len(block.text)
        
        notes["layout"]["page_count"] = len(page_numbers)
        notes["content"]["total_text_length"] = total_text_length
        
        return notes
    
    def extract_text_with_spans(self, blocks: List[DocumentBlock]) -> str:
        """
        Extract full text with character spans.
        
        Args:
            blocks: Parsed document blocks
            
        Returns:
            Full text with span information
        """
        full_text = ""
        spans = []
        
        for i, block in enumerate(blocks):
            start_pos = len(full_text)
            block_text = block.text + "\n"
            full_text += block_text
            end_pos = len(full_text)
            
            spans.append(EvidenceSpan(
                start=start_pos,
                end=end_pos,
                text=block.text
            ))
        
        return full_text.strip()
    
    def extract_tables(self, blocks: List[DocumentBlock]) -> List[Dict[str, Any]]:
        """
        Extract tables from document blocks.
        
        Args:
            blocks: Parsed document blocks
            
        Returns:
            List of table data
        """
        tables = []
        
        for block in blocks:
            if block.block_type == "table":
                table_data = {
                    "text": block.text,
                    "page_number": block.page_number,
                    "metadata": block.metadata
                }
                
                # Try to extract structured table data
                if hasattr(block.element, 'metadata') and 'text_as_html' in block.element.metadata:
                    table_data['html'] = block.element.metadata['text_as_html']
                
                tables.append(table_data)
        
        return tables
    
    def extract_figures(self, blocks: List[DocumentBlock]) -> List[Dict[str, Any]]:
        """
        Extract figures from document blocks.
        
        Args:
            blocks: Parsed document blocks
            
        Returns:
            List of figure data
        """
        figures = []
        
        for block in blocks:
            if block.block_type == "figure":
                figure_data = {
                    "text": block.text,
                    "page_number": block.page_number,
                    "metadata": block.metadata
                }
                
                # Extract figure caption if available
                if hasattr(block.element, 'metadata') and 'caption' in block.element.metadata:
                    figure_data['caption'] = block.element.metadata['caption']
                
                figures.append(figure_data)
        
        return figures
    
    def extract_formulas(self, blocks: List[DocumentBlock]) -> List[Dict[str, Any]]:
        """
        Extract formulas from document blocks.
        
        Args:
            blocks: Parsed document blocks
            
        Returns:
            List of formula data
        """
        formulas = []
        
        for block in blocks:
            if block.block_type == "formula":
                formula_data = {
                    "text": block.text,
                    "page_number": block.page_number,
                    "metadata": block.metadata
                }
                
                formulas.append(formula_data)
        
        return formulas
    
    def save_parsed_document(self, parsed_data: Dict[str, Any], output_path: Path) -> None:
        """
        Save parsed document data to file.
        
        Args:
            parsed_data: Parsed document data
            output_path: Output file path
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            
            gola_logger.info(f"Saved parsed document to: {output_path}")
        
        except Exception as e:
            gola_logger.error(f"Error saving parsed document to {output_path}: {e}")
            raise
    
    def load_parsed_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Load parsed document data from file.
        
        Args:
            file_path: Path to parsed document file
            
        Returns:
            Parsed document data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        except Exception as e:
            gola_logger.error(f"Error loading parsed document from {file_path}: {e}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats.
        
        Returns:
            List of supported file extensions
        """
        if not UNSTRUCTURED_AVAILABLE:
            return []
        
        return [
            '.txt', '.md', '.pdf', '.docx', '.doc', '.html', '.htm',
            '.xml', '.json', '.csv', '.rtf', '.odt', '.epub'
        ]
    
    def is_format_supported(self, file_path: Path) -> bool:
        """
        Check if document format is supported.
        
        Args:
            file_path: Path to document
            
        Returns:
            True if format is supported
        """
        return file_path.suffix.lower() in self.get_supported_formats()

# Global document parser instance
document_parser = DocumentParser()

def get_document_parser() -> DocumentParser:
    """Get the global document parser instance."""
    return document_parser

def init_document_parser(ocr_enabled: bool = True, ocr_language: str = 'eng') -> DocumentParser:
    """
    Initialize the global document parser.
    
    Args:
        ocr_enabled: Enable OCR
        ocr_language: OCR language
        
    Returns:
        Document parser instance
    """
    global document_parser
    document_parser = DocumentParser(ocr_enabled, ocr_language)
    return document_parser 