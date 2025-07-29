"""
Lazy loader utilities for JSONL-based glyph, animation, and icon data.

This module provides efficient streaming and search capabilities for the modular
glyph system, supporting accessibility-first design and agent/LLM workflows.
"""

import json
import re
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Union, Set

def stream_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Stream records from a JSONL file, yielding one record at a time.
    
    Args:
        path: Path to the JSONL file
        
    Yields:
        Dict containing the parsed JSON record
        
    Example:
        >>> for glyph in stream_jsonl("assets/glyphs.jsonl"):
        ...     print(glyph["glyph_id"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")

def find_by_tag(path: Union[str, Path], tag: str) -> List[Dict[str, Any]]:
    """
    Find all records containing a specific tag.
    
    Args:
        path: Path to the JSONL file
        tag: Tag to search for
        
    Returns:
        List of matching records
        
    Example:
        >>> circle_glyphs = find_by_tag("assets/glyphs.jsonl", "circle")
    """
    return [rec for rec in stream_jsonl(path) if tag in rec.get('tags', [])]

def find_by_accessibility_label(path: Union[str, Path], label: str) -> Optional[Dict[str, Any]]:
    """
    Find a record by its accessibility label (case-insensitive).
    
    Args:
        path: Path to the JSONL file
        label: Accessibility label to search for
        
    Returns:
        Matching record or None if not found
        
    Example:
        >>> glyph = find_by_accessibility_label("assets/glyphs.jsonl", "Circle glyph")
    """
    label_lower = label.lower()
    for rec in stream_jsonl(path):
        if rec.get('accessibility_label', '').lower() == label_lower:
            return rec
    return None

def find_by_id(path: Union[str, Path], id_field: str, value: str) -> Optional[Dict[str, Any]]:
    """
    Find a record by a specific ID field.
    
    Args:
        path: Path to the JSONL file
        id_field: Name of the ID field (e.g., 'glyph_id', 'icon_id', 'animation_id')
        value: Value to search for
        
    Returns:
        Matching record or None if not found
        
    Example:
        >>> glyph = find_by_id("assets/glyphs.jsonl", "glyph_id", "U+E200")
    """
    for rec in stream_jsonl(path):
        if rec.get(id_field) == value:
            return rec
    return None

def count_records(path: Union[str, Path]) -> int:
    """
    Count the number of records in a JSONL file.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        Number of records
    """
    return sum(1 for _ in stream_jsonl(path))

def get_all_tags(path: Union[str, Path]) -> Set[str]:
    """
    Get all unique tags from a JSONL file.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        Set of unique tags
    """
    tags = set()
    for rec in stream_jsonl(path):
        tags.update(rec.get('tags', []))
    return tags

def search_by_regex(path: Union[str, Path], field: str, pattern: str) -> List[Dict[str, Any]]:
    """
    Search records by regex pattern on a specific field.
    
    Args:
        path: Path to the JSONL file
        field: Field name to search in
        pattern: Regex pattern to match
        
    Returns:
        List of matching records
        
    Example:
        >>> circle_glyphs = search_by_regex("assets/glyphs.jsonl", "name", r"circle")
    """
    regex = re.compile(pattern, re.IGNORECASE)
    return [rec for rec in stream_jsonl(path) if regex.search(str(rec.get(field, '')))]

def find_by_multiple_tags(path: Union[str, Path], tags: List[str], match_all: bool = True) -> List[Dict[str, Any]]:
    """
    Find records that match multiple tags.
    
    Args:
        path: Path to the JSONL file
        tags: List of tags to search for
        match_all: If True, record must contain all tags. If False, any tag.
        
    Returns:
        List of matching records
        
    Example:
        >>> basic_shapes = find_by_multiple_tags("assets/glyphs.jsonl", ["shape", "basic"], match_all=True)
    """
    results = []
    for rec in stream_jsonl(path):
        record_tags = set(rec.get('tags', []))
        if match_all:
            if all(tag in record_tags for tag in tags):
                results.append(rec)
        else:
            if any(tag in record_tags for tag in tags):
                results.append(rec)
    return results 