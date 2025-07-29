"""
Core data models and schemas for the Perspective D<cide> framework.

Defines the fundamental data structures used throughout the framework
for content processing, categorization, and analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

class AnalysisType(str, Enum):
    """Types of analysis supported by the framework."""
    CATEGORIZATION = "categorization"
    SENTIMENT = "sentiment"
    ENTITY_EXTRACTION = "entity_extraction"
    TOPIC_MODELING = "topic_modeling"
    SYMBOLIC_REASONING = "symbolic_reasoning"  # Our glyph system
    TAROT_MAPPING = "tarot_mapping"  # Our tarot system

class ContentItem(BaseModel):
    """Represents a piece of content for analysis."""
    
    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., description="Content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CategoryProposal(BaseModel):
    """A proposed category for content."""
    
    category_name: str = Field(..., description="Category name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Reasoning for this category")
    agent_id: str = Field(..., description="Agent that proposed this category")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    description: str = Field("", description="Category description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v

class AnalysisResult(BaseModel):
    """Result from content analysis."""
    
    content_id: str = Field(..., description="Content identifier")
    analysis_type: AnalysisType = Field(..., description="Type of analysis")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

class SymbolicAnalysisResult(AnalysisResult):
    """Result from symbolic analysis using glyphs and tarot."""
    
    glyphs_used: List[str] = Field(default_factory=list, description="Glyphs used in analysis")
    tarot_cards: List[str] = Field(default_factory=list, description="Tarot cards referenced")
    collapse_state: Dict[str, Any] = Field(default_factory=dict, description="Symbolic collapse state")
    archetypal_theme: Optional[str] = Field(None, description="Archetypal theme identified")

class ProcessingPipeline(BaseModel):
    """Configuration for a processing pipeline."""
    
    name: str = Field(..., description="Pipeline name")
    steps: List[str] = Field(..., description="Processing steps")
    config: Dict[str, Any] = Field(default_factory=dict, description="Pipeline configuration")
    enabled: bool = Field(True, description="Whether pipeline is enabled")

class FrameworkMetrics(BaseModel):
    """Framework performance and usage metrics."""
    
    content_processed: int = Field(0, description="Number of content items processed")
    categories_discovered: int = Field(0, description="Number of categories discovered")
    processing_time: float = Field(0.0, description="Total processing time in seconds")
    error_count: int = Field(0, description="Number of errors encountered")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp") 