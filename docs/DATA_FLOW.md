# Gola Data Flow Documentation

## Overview

This document describes how data flows through the Gola system, from initial ingestion to final output. The system is designed with clear separation of concerns, robust error handling, and comprehensive quality assurance.

## High-Level Data Flow

```
Input Sources → Ingestion → Processing → Building → Validation → Export
     │            │           │          │         │           │
     ▼            ▼           ▼          ▼         ▼           ▼
[Files/Web] → [Parse/OCR] → [Chunk] → [AI] → [Validate] → [JSONL/CSV/Parquet]
     │            │           │          │         │           │
     ▼            ▼           ▼          ▼         ▼           ▼
[LLM-Ready] → [Layout] → [Dedup] → [Router] → [Cross-Check] → [Vector DB]
```

## Detailed Data Flow

### 1. Input Sources

#### File System Input
```
User Command: corpusctl ingest --source ./library
     │
     ▼
[File Watcher] → [Manifest DB] → [File Processor]
     │                │               │
     ▼                ▼               ▼
[New/Changed Files] → [SHA256 Hash] → [Format Detection]
```

#### Web Input
```
User Command: corpusctl crawl plan --site https://example.com
     │
     ▼
[LLM-Ready Probe] → [Crawl4AI] → [Clean Markdown]
     │                    │              │
     ▼                    ▼              ▼
[/llms.txt, .md] → [HTML Scraping] → [Document Parser]
```

#### API Input
```
User Command: corpusctl ingest --api-config api.yaml
     │
     ▼
[API Client] → [Rate Limiting] → [Data Normalization]
     │              │                    │
     ▼              ▼                    ▼
[HTTP Requests] → [Response Cache] → [Format Conversion]
```

### 2. Ingestion Pipeline

#### Document Parsing
```
Raw Document → [unstructured] → [OCR (if needed)] → [Layout Analysis]
     │              │                │                    │
     ▼              ▼                ▼                    ▼
[PDF/HTML/DOCX] → [Blocks] → [Text Extraction] → [Page Structure]
```

#### LLM-Ready Detection
```
Web URL → [Probe Sequence] → [Asset Selection] → [Content Fetch]
     │           │                │                │
     ▼           ▼                ▼                ▼
[Site Root] → [Priority Check] → [Best Source] → [Markdown/JSON]
```

#### Manifest Management
```
File/URL → [Metadata Extraction] → [Hash Generation] → [Database Record]
     │              │                    │                │
     ▼              ▼                    ▼                ▼
[Source Info] → [File Properties] → [SHA256] → [Manifest Entry]
```

### 3. Processing Pipeline

#### Text Normalization
```
Raw Text → [Encoding Fix] → [Whitespace Clean] → [Hyphenation Fix]
     │           │                │                    │
     ▼           ▼                ▼                    ▼
[Extracted] → [UTF-8] → [Normalized] → [Unwrapped Lines]
```

#### Content-Defined Chunking
```
Normalized Text → [FastCDC] → [Paragraph Detection] → [Stable Boundaries]
       │              │              │                    │
       ▼              ▼              ▼                    ▼
[Full Document] → [Chunks] → [Paragraph Units] → [Chunk Hash]
```

#### Deduplication
```
Chunks → [SimHash] → [MinHash+LSH] → [Vector Similarity] → [Dedup Decision]
   │         │            │                │                    │
   ▼         ▼            ▼                ▼                    ▼
[Text] → [Fingerprint] → [LSH Bands] → [Cosine Distance] → [Keep/Drop]
```

### 4. Building Pipeline

#### Task Routing
```
Chunk + Task → [GPU Monitor] → [Router Decision] → [Model Selection]
     │              │                │                    │
     ▼              ▼                ▼                    ▼
[Processing Request] → [Resource Check] → [Policy Apply] → [Provider+Model]
```

#### AI Processing
```
Chunk → [Prompt Generation] → [Model Call] → [Response Parse] → [Schema Validation]
   │            │                │              │                    │
   ▼            ▼                ▼              ▼                    ▼
[Text] → [Task-Specific] → [API Request] → [JSON Response] → [Pydantic Model]
```

#### Evidence Span Extraction
```
Model Response → [Span Detection] → [Source Mapping] → [Validation]
       │              │                │                │
       ▼              ▼                ▼                ▼
[AI Output] → [Character Offsets] → [Source Text] → [Span Verification]
```

### 5. Validation Pipeline

#### Schema Validation
```
Raw Output → [Pydantic Schema] → [Type Check] → [Constraint Validation]
      │              │                │                │
      ▼              ▼                ▼                ▼
[AI Response] → [Schema Definition] → [Type Safety] → [Valid/Invalid]
```

#### Cross-Validation
```
Output A → [Model B] → [Comparison] → [Agreement Check] → [Final Decision]
     │         │            │                │                │
     ▼         ▼            ▼                ▼                ▼
[Primary] → [Secondary] → [Similarity] → [Consensus] → [Adjudicated]
```

#### Quality Gates
```
Validated Output → [Coverage Check] → [Dup Check] → [Hallucination Check] → [Quality Score]
        │                │                │                │                    │
        ▼                ▼                ▼                ▼                    ▼
[Schema Valid] → [Evidence %] → [Duplicate %] → [Claim Check] → [Pass/Fail]
```

### 6. Export Pipeline

#### Format Generation
```
Validated Data → [JSONL Writer] → [CSV Flattener] → [Parquet Converter]
       │              │                │                    │
       ▼              ▼                ▼                    ▼
[Structured] → [Line-by-Line] → [Tabular] → [Columnar]
```

#### Vector Indexing
```
Chunks + Embeddings → [Qdrant Client] → [Collection Update] → [Index Optimization]
         │                    │                │                    │
         ▼                    ▼                ▼                    ▼
[Text + Vectors] → [Vector DB] → [Payload Storage] → [Search Index]
```

#### Provenance Tracking
```
All Data → [Hash Generation] → [Manifest Creation] → [Audit Trail]
     │            │                │                    │
     ▼            ▼                ▼                    ▼
[Processed] → [Content Hash] → [Metadata] → [Complete History]
```

## Data Structures

### Input Data
```python
# File Input
{
    "file_id": "uuid",
    "path": "/path/to/file.pdf",
    "file_sha256": "abc123...",
    "size_bytes": 1024000,
    "mtime_utc": "2024-12-19T10:00:00Z",
    "mime_type": "application/pdf",
    "language": "en",
    "pii_level": 0
}

# Web Input
{
    "page_id": "uuid",
    "url": "https://example.com/page",
    "canonical_url": "https://example.com/page",
    "content_sha256": "def456...",
    "html_path": "/data/raw/def456.html",
    "md_clean_path": "/data/parsed/def456_clean.md",
    "md_fit_path": "/data/parsed/def456_fit.md",
    "source_kind": "llms"  # llms, md, plugin, html
}
```

### Processed Data
```python
# Chunk Data
{
    "chunk_hash": "ghi789...",
    "file_id": "uuid",
    "char_start": 18420,
    "char_end": 18796,
    "text_norm": "Normalized paragraph text...",
    "simhash64": "a3f1...",
    "minhash_sig": ["...", "...", "..."],
    "duplicate_of": null
}

# Run Data
{
    "run_id": "uuid",
    "chunk_hash": "ghi789...",
    "task_type": "summary",
    "prompt_version": "1.0",
    "model_id": "claude-3.7-sonnet",
    "provider": "anthropic",
    "status": "ok",
    "token_in": 512,
    "token_out": 256,
    "cost_usd": 0.002,
    "job_key": "unique_hash"
}
```

### Output Data
```python
# Summary Output
{
    "run_id": "uuid",
    "summary_text": "The document discusses...",
    "keypoints": ["Point 1", "Point 2"],
    "evidence_spans": [
        {"start": 18430, "end": 18495},
        {"start": 18610, "end": 18670}
    ],
    "confidence": 0.95
}

# Entity Output
{
    "run_id": "uuid",
    "entity_text": "Solana",
    "entity_type": "Blockchain",
    "start": 18510,
    "end": 18516,
    "confidence": 0.98
}
```

## Quality Assurance Flow

### 1. Evidence Span Validation
```
Claim: "The document discusses blockchain technology"
     │
     ▼
[Span Check] → [Source Verification] → [Coverage Analysis]
     │                │                    │
     ▼                ▼                    ▼
[Character Offsets] → [Source Text Match] → [Coverage %]
```

### 2. Cross-Model Validation
```
Model A Output → [Model B Check] → [Model C Check] → [Consensus]
       │                │                │                │
       ▼                ▼                ▼                ▼
[Primary Result] → [Secondary] → [Tertiary] → [Final Decision]
```

### 3. Quality Metrics
```
Dataset → [Coverage Analysis] → [Duplication Check] → [Hallucination Detection]
    │              │                    │                        │
    ▼              ▼                    ▼                        ▼
[All Outputs] → [Evidence %] → [Duplicate %] → [Unsupported Claims %]
```

## Error Handling

### 1. Retry Logic
```
API Call → [Success] → [Return Result]
     │
     ▼
[Failure] → [Backoff] → [Retry] → [Max Retries] → [Fallback]
```

### 2. Circuit Breaker
```
Provider → [Failure Count] → [Threshold Check] → [Open Circuit] → [Timeout] → [Close]
     │              │                │                │              │         │
     ▼              ▼                ▼                ▼              ▼         ▼
[API Calls] → [Error Tracking] → [Failure Rate] → [Block Calls] → [Wait] → [Reset]
```

### 3. Graceful Degradation
```
Task → [Primary Model] → [Failure] → [Secondary Model] → [Failure] → [Local Model]
   │            │              │                │              │            │
   ▼            ▼              ▼                ▼              ▼            ▼
[Request] → [Cloud API] → [Error] → [Different Cloud] → [Error] → [LM Studio]
```

## Performance Optimization

### 1. Batch Processing
```
Chunks → [Batch Grouping] → [Parallel Processing] → [Result Aggregation]
    │              │                    │                    │
    ▼              ▼                    ▼                    ▼
[Individual] → [Batches] → [Concurrent] → [Combined]
```

### 2. Caching Strategy
```
Request → [Cache Check] → [Hit] → [Return Cached]
     │            │              │              │
     ▼            ▼              ▼              ▼
[New Request] → [Hash Lookup] → [Miss] → [Process] → [Cache Result]
```

### 3. Resource Management
```
GPU Monitor → [Utilization Check] → [Threshold] → [Load Balancing]
      │                │                │                │
      ▼                ▼                ▼                ▼
[Real-time] → [Resource %] → [Over Limit] → [Route to Cloud]
```

## Data Persistence

### 1. Database Schema
```sql
-- Core tables for data tracking
files (file_id, path, file_sha256, size_bytes, mtime_utc, mime_type, language, pii_level)
chunks (chunk_hash, file_id, char_start, char_end, text_norm, simhash64, minhash_sig)
runs (run_id, chunk_hash, task_type, prompt_version, model_id, provider, status, cost_usd)
outputs_* (run_id, task_specific_columns, evidence_json)
```

### 2. Vector Database
```python
# Qdrant Collections
{
    "name": "corpus_chunks_v1",
    "vector_size": 768,
    "distance": "Cosine",
    "payload": {
        "file_sha256": "abc123...",
        "chunk_hash": "ghi789...",
        "pii_level": 0,
        "topics": ["blockchain", "technology"],
        "mode": "general"
    }
}
```

### 3. File Storage
```
data/
├── raw/           # Original files
├── parsed/        # Processed documents
├── notes/         # Layout analysis
├── chunks/        # Normalized chunks
├── outputs/       # Task results
└── manifests/     # Provenance data
```

This data flow ensures that Gola maintains data integrity, provides comprehensive quality assurance, and delivers reliable, validated datasets for downstream use. 