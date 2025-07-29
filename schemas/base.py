"""
Base schemas for the Gola system.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from pydantic import BaseModel, Field, validator

class EvidenceSpan(BaseModel):
    """Evidence span with character-level references."""
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")
    text: Optional[str] = Field(None, description="Extracted text from span")
    
    @validator('end')
    def end_must_be_after_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError('end must be after start')
        return v

class FileMetadata(BaseModel):
    """File metadata information."""
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    path: Path = Field(..., description="File path")
    file_sha256: str = Field(..., description="SHA256 hash of file")
    size_bytes: int = Field(..., description="File size in bytes")
    mtime_utc: datetime = Field(..., description="File modification time")
    mime_type: Optional[str] = Field(None, description="MIME type")
    language: Optional[str] = Field(None, description="Detected language")
    pii_level: int = Field(0, ge=0, le=3, description="PII sensitivity level")
    added_at_utc: datetime = Field(default_factory=datetime.utcnow)

class ChunkMetadata(BaseModel):
    """Chunk metadata information."""
    chunk_hash: str = Field(..., description="Chunk hash")
    file_id: str = Field(..., description="Source file ID")
    char_start: int = Field(..., description="Start character position")
    char_end: int = Field(..., description="End character position")
    text_norm: str = Field(..., description="Normalized text content")
    simhash64: Optional[str] = Field(None, description="SimHash fingerprint")
    minhash_sig: Optional[bytes] = Field(None, description="MinHash signature")
    duplicate_of: Optional[str] = Field(None, description="Duplicate chunk hash")

class RunMetadata(BaseModel):
    """Run metadata for AI processing."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_hash: str = Field(..., description="Source chunk hash")
    task_type: str = Field(..., description="Task type (summary, entities, etc.)")
    prompt_version: str = Field(..., description="Prompt version")
    model_id: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Model provider")
    status: str = Field("ok", description="Run status (ok, retry, failed)")
    started_at_utc: Optional[datetime] = Field(None)
    finished_at_utc: Optional[datetime] = Field(None)
    token_in: Optional[int] = Field(None, description="Input tokens")
    token_out: Optional[int] = Field(None, description="Output tokens")
    cost_usd: Optional[float] = Field(None, description="API cost in USD")
    job_key: str = Field(..., description="Unique job key")

class QualityMetrics(BaseModel):
    """Quality metrics for outputs."""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    coverage: float = Field(..., ge=0.0, le=1.0, description="Coverage ratio")
    duplication_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Duplication ratio")
    hallucination_risk: float = Field(0.0, ge=0.0, le=1.0, description="Hallucination risk")
    evidence_span_coverage: float = Field(..., ge=0.0, le=1.0, description="Evidence span coverage")
    
    @validator('confidence', 'coverage', 'evidence_span_coverage')
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Probability values must be between 0.0 and 1.0')
        return v

class BaseOutput(BaseModel):
    """Base class for all outputs."""
    run_id: str = Field(..., description="Run ID")
    quality_metrics: QualityMetrics = Field(..., description="Quality metrics")
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list, description="Evidence spans")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }

class DatasetManifest(BaseModel):
    """Dataset manifest information."""
    dataset_slug: str = Field(..., description="Dataset identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    mode: str = Field(..., description="Processing mode")
    total_files: int = Field(0, description="Total files processed")
    total_chunks: int = Field(0, description="Total chunks generated")
    total_runs: int = Field(0, description="Total AI runs")
    total_cost_usd: float = Field(0.0, description="Total cost in USD")
    quality_metrics: QualityMetrics = Field(..., description="Overall quality metrics")
    file_hashes: List[str] = Field(default_factory=list, description="File SHA256 hashes")
    chunk_hashes: List[str] = Field(default_factory=list, description="Chunk hashes")
    run_ids: List[str] = Field(default_factory=list, description="Run IDs")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PlanSpec(BaseModel):
    """Planning specification for data processing."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_slug: str = Field(..., description="Target dataset")
    mode: str = Field(..., description="Processing mode")
    source_path: Path = Field(..., description="Source data path")
    tasks: List[str] = Field(..., description="Tasks to perform")
    budget_usd: float = Field(..., description="Budget in USD")
    router_policy: str = Field("throughput", description="Router policy")
    chunking_config: Dict[str, Any] = Field(default_factory=dict)
    dedup_config: Dict[str, Any] = Field(default_factory=dict)
    validation_config: Dict[str, Any] = Field(default_factory=dict)
    export_config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = Field(None, description="Planning notes")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }

class GPUStatus(BaseModel):
    """GPU status information."""
    gpu_index: int = Field(..., description="GPU index")
    utilization: float = Field(..., ge=0.0, le=100.0, description="GPU utilization %")
    memory_used_mb: int = Field(..., description="Memory used in MB")
    memory_total_mb: int = Field(..., description="Total memory in MB")
    temperature_c: Optional[float] = Field(None, description="Temperature in Celsius")
    power_w: Optional[float] = Field(None, description="Power consumption in Watts")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage."""
        return (self.memory_used_mb / self.memory_total_mb) * 100.0
    
    @property
    def is_overloaded(self) -> bool:
        """Check if GPU is overloaded."""
        return self.utilization > 85.0 or self.memory_utilization > 92.0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProviderConfig(BaseModel):
    """Provider configuration."""
    name: str = Field(..., description="Provider name")
    enabled: bool = Field(True, description="Provider enabled")
    base_url: Optional[str] = Field(None, description="Base URL")
    api_key: Optional[str] = Field(None, description="API key")
    models: List[str] = Field(default_factory=list, description="Available models")
    max_tokens: int = Field(4096, description="Maximum tokens")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Temperature")
    cost_per_1k_tokens: Optional[float] = Field(None, description="Cost per 1K tokens")
    rate_limit_rpm: Optional[int] = Field(None, description="Rate limit requests per minute")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class RouterDecision(BaseModel):
    """Router decision for model selection."""
    task_type: str = Field(..., description="Task type")
    chunk_hash: str = Field(..., description="Chunk hash")
    selected_provider: str = Field(..., description="Selected provider")
    selected_model: str = Field(..., description="Selected model")
    reasoning: str = Field(..., description="Decision reasoning")
    gpu_status: Optional[GPUStatus] = Field(None, description="GPU status at decision time")
    pii_level: int = Field(0, ge=0, le=3, description="PII level")
    estimated_cost: float = Field(0.0, description="Estimated cost")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 