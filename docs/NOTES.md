# Gola - AI Swarm System for Document Processing

## Implementation Progress

### ✅ **Phase 1: Core Pipeline Components (COMPLETED)**

**Status**: Core pipeline components implemented and tested

#### **Components Built:**

1. **CLI Utilities** (`cli/utils/`)
   - ✅ `config.py` - Configuration loading and validation
   - ✅ `logging.py` - Rich logging with file and console output

2. **Base Schemas** (`schemas/`)
   - ✅ `base.py` - Core data structures (FileMetadata, ChunkMetadata, RunMetadata, etc.)

3. **GPU Monitoring** (`pipeline/monitoring/`)
   - ✅ `gpu.py` - NVML-based GPU monitoring with utilization, memory, temperature tracking

4. **File Ingestion** (`pipeline/ingest/`)
   - ✅ `watcher.py` - File system watcher with event handling and metadata extraction
   - ✅ `parser.py` - Document parser using unstructured library with OCR support

5. **Text Processing** (`pipeline/normalize/`)
   - ✅ `chunker.py` - Content-defined chunking (FastCDC) with paragraph-aware segmentation
   - ✅ `dedup.py` - Multi-method deduplication (SimHash, MinHash, vector similarity)

6. **LLM Routing** (`pipeline/router/`)
   - ✅ `llm_router.py` - Intelligent model selection with GPU-aware routing policies

7. **Testing**
   - ✅ `test_core_components.py` - End-to-end test script for all core components

#### **Key Features Implemented:**

- **GPU Monitoring**: Real-time GPU utilization, memory, temperature, and power monitoring
- **File Watching**: Automatic detection of new/changed files with metadata extraction
- **Document Parsing**: Support for PDF, DOCX, HTML, TXT, JSON with OCR capabilities
- **Content Chunking**: FastCDC-based chunking with semantic boundary preservation
- **Deduplication**: Multi-layer deduplication using SimHash, MinHash, and vector similarity
- **LLM Routing**: Intelligent provider selection based on GPU status, task type, and policies
- **Rich Logging**: Structured logging with file and console output
- **Configuration Management**: YAML-based configuration with environment variable support

#### **Test Results:**
```bash
python test_core_components.py
# Expected output: All core components working with test document processing
```

### ✅ **Phase 2: AI Builders & Validation (COMPLETED)**

**Status**: AI builders and validation system implemented

#### **Components Built:**

1. **AI Builders** (`pipeline/builders/`)
   - ✅ `base.py` - Abstract base class for all AI builders
   - ✅ `summary_builder.py` - Document summarization with extractive and abstractive methods
   - ✅ `entities_builder.py` - Named entity extraction with confidence scoring

2. **Validation** (`pipeline/validate/`)
   - ✅ `validator.py` - Output validation with Pydantic schemas, quality assessment, and multi-model consensus

#### **Key Features Implemented:**

- **Base Builder Class**: Common functionality for LLM routing, caching, and validation
- **Summary Builder**: Extractive summaries with bullet points and abstractive summaries
- **Entities Builder**: Named entity recognition with entity types and confidence scores
- **Validation System**: Schema validation, quality metrics, and cross-model verification
- **JSON Schema Validation**: Type-safe outputs with Pydantic models
- **Evidence Spans**: Character-level evidence tracking for auditability

### ✅ **Phase 3: CLI & MCP Integration (COMPLETED)**

**Status**: CLI framework and commands fully implemented

#### **Components Built:**

1. **CLI Framework** (`cli/`)
   - ✅ `main.py` - Main CLI entry point with Typer integration
   - ✅ `utils/config.py` - Configuration management
   - ✅ `utils/logging.py` - Logging setup
   - ✅ `commands/__init__.py` - Command package initialization

2. **CLI Commands** (`cli/commands/`)
   - ✅ `plan.py` - Planning and scouting operations with risk assessment
   - ✅ `ingest.py` - File ingestion with watching and validation
   - ✅ `build.py` - AI processing and dataset building
   - ✅ `export.py` - Data export and vectorization
   - ✅ `validate.py` - Data validation and quality assessment (placeholder)
   - ✅ `crawl.py` - Web crawling commands (placeholder)
   - ✅ `mcp.py` - MCP server commands (placeholder)
   - ✅ `hub.py` - Hub integration commands

#### **Key Features Implemented:**

- **Comprehensive CLI**: Full command-line interface with subcommands
- **Planning System**: Risk assessment, cost estimation, and recommendations
- **Ingestion Pipeline**: File processing with deduplication and validation
- **Dataset Building**: AI-powered processing with multiple tasks
- **Export System**: Multiple format export with vectorization
- **Status Monitoring**: Real-time status tracking for all operations
- **Rich Output**: Beautiful tables and progress indicators
- **Error Handling**: Comprehensive error handling and logging

#### **CLI Usage Examples:**
```bash
# Plan a processing operation
corpusctl plan create ./documents --mode fiction --budget 50 --verbose

# Ingest files with watching
corpusctl ingest files ./documents --watch --recursive

# Build dataset with AI processing
corpusctl build dataset ./data/ingested --tasks summary,entities --parallel 8

# Export in multiple formats
corpusctl export dataset ./data/datasets/dataset.jsonl --formats jsonl,csv,parquet --vectorize

# Check system status
corpusctl status
```

### ✅ **Liber777revised JP2 Dataset Processing (COMPLETED)**

**Status**: Comprehensive dataset processor implemented and ready for use

#### **Components Built:**

1. **Liber777 Dataset Processor** (`liber777_dataset_processor.py`)
   - ✅ Full dataset analysis of 170 JP2 files
   - ✅ Content type detection and classification
   - ✅ Metadata extraction and organization
   - ✅ Multiple export formats (CSV, JSON, Excel, YAML)
   - ✅ Quality assessment and validation
   - ✅ Dataset statistics and insights

2. **Test Script** (`test_liber777_processor.py`)
   - ✅ Sample processing with user interaction
   - ✅ Full dataset processing option
   - ✅ Progress tracking and error handling

3. **Documentation** (`docs/LIBER777_USAGE_GUIDE.md`)
   - ✅ Comprehensive usage guide
   - ✅ Installation instructions
   - ✅ Troubleshooting guide
   - ✅ Advanced usage examples

#### **Key Features Implemented:**

- **Image Analysis**: PIL and OpenCV integration for content detection
- **Content Classification**: Text, image, mixed content identification
- **Quality Metrics**: Corruption detection, readability scoring
- **Page Analysis**: Sequential numbering, document type classification
- **Multiple Exports**: CSV, Excel, JSON, YAML, and text report formats
- **Error Handling**: Graceful handling of corrupted or problematic files
- **Progress Tracking**: Real-time progress bars and comprehensive logging

#### **Usage:**
```bash
# Quick test with sample files
python test_liber777_processor.py

# Full dataset processing
python liber777_dataset_processor.py

# Custom processing
python liber777_dataset_processor.py --dataset-path "C:\Users\thoma\Desktop\Liber777revised_jp2" --output-dir "my_analysis" --format all
```

#### **Output Files Generated:**
- `liber777_dataset_analysis.csv` - Main analysis data
- `liber777_dataset_analysis.xlsx` - Excel workbook with metadata
- `liber777_dataset_analysis.json` - Structured JSON data
- `liber777_metadata.yaml` - YAML metadata
- `liber777_summary_report.txt` - Human-readable report
- `liber777_processing.log` - Processing log

### ✅ **Symbolic Extraction System (COMPLETED)**

**Status**: Advanced symbolic extraction system for Liber777revised implemented

#### **Components Built:**

1. **Symbolic Extractor** (`liber777_symbolic_extractor.py`)
   - ✅ Comprehensive symbolic knowledge base (planetary, elemental, qabalistic, zodiacal, numerical, color)
   - ✅ Pattern recognition for geometric shapes, numerical sequences, and textual patterns
   - ✅ Correspondence extraction from text, visual elements, and tables
   - ✅ Relationship generation and symbolic dataset creation
   - ✅ Training/validation/test dataset generation for LoRA training

2. **Test System** (`test_symbolic_extraction.py`)
   - ✅ Dummy data generation for testing
   - ✅ Symbolic knowledge loading and validation
   - ✅ Pattern recognition testing
   - ✅ Correspondence extraction validation
   - ✅ Training dataset creation and saving

3. **Documentation**
   - ✅ `LIBER777_SYMBOLIC_TRAINING_GUIDE.md` - Comprehensive usage guide
   - ✅ `SYMBOLIC_EXTRACTION_SUMMARY.md` - System overview and architecture

#### **Key Features Implemented:**

- **Symbolic Knowledge Base**: 7 planetary, 4 elemental, 10 qabalistic, 12 zodiacal, 10 numerical, 8 color correspondences
- **Pattern Recognition**: Geometric shapes, numerical sequences, textual patterns
- **Correspondence Extraction**: Text-based, visual, pattern-based, and table-based extraction
- **Relationship Generation**: Automatic generation of symbolic relationships and associations
- **LoRA Training Data**: JSONL format training pairs for fine-tuning models on symbolic reasoning
- **Quality Assessment**: Confidence scoring and validation for extracted correspondences

#### **Usage:**
```bash
# Test symbolic extraction
python test_symbolic_extraction.py

# Extract symbolic data from Liber777revised
python liber777_symbolic_extractor.py

# Review generated datasets
ls symbolic_datasets/
```

### 🔄 **Phase 4: Framework-First Architecture (IN PROGRESS)**

**Status**: Transitioning to pip-installable framework approach

#### **Framework Components:**

1. **Core Framework** (`perspective-dcide`)
   - [ ] `__init__.py` - Main package initialization
   - [ ] `core/` - Core framework components
   - [ ] `etx/` - Emergent TaXonomy system
   - [ ] `builders/` - AI builder components
   - [ ] `validators/` - Validation components
   - [ ] `utils/` - Utility functions

2. **Emergent TaXonomy (ETX)** (`etx/`)
   - [ ] `taxonomy.py` - Dynamic taxonomy generation
   - [ ] `clustering.py` - Content clustering algorithms
   - [ ] `embeddings.py` - Embedding generation and management
   - [ ] `storage.py` - SQLite-based taxonomy storage

3. **Documentation**
   - ✅ `docs/PDEC_ETX_FRAMEWORK.md` - Framework documentation

### 🔄 **Phase 5: Web Scraping & Hub Integration (PLANNED)**

**Status**: Architecture ready

#### **Components to Build:**

1. **Web Scraping** (`pipeline/web/`)
   - [ ] `crawler.py` - Crawl4AI integration
   - [ ] `llm_ready.py` - LLM-ready content detection

2. **Hub Integration** (`hub/`)
   - ✅ Already implemented in previous sessions

### 🔄 **Phase 6: Future States & Symbolic Reasoning (IN PROGRESS)**

**Status**: Conceptual framework developed, implementation in progress

#### **Core Concept:**
The system is evolving to leverage symbolic extraction capabilities to build JSONL datasets representing "different future states of things" - from GUI elements to colors rendering on screen. This involves:

1. **Symbolic Representation**: Extracting abstract meanings and relationships from content
2. **Future State Modeling**: LLM generation of "likely events" based on file usage patterns
3. **System Hashing**: Hashing entire file systems to understand current state
4. **Placeholder Scaffolding**: Using placeholders for undefined code (functions/OOP)
5. **Agent Scaffolding**: Allowing coding agents to scaffold processes using symbolic JSONL "flags of potential"
6. **Recursive Collapse**: Collapsing all potentials cheaply, easily, and rigidly for symbolic reasoning

#### **Components to Build:**

1. **Symbolic State Extractor** (`symbolic/`)
   - [ ] `state_extractor.py` - Extract current system states
   - [ ] `future_predictor.py` - Predict future states
   - [ ] `potential_collapser.py` - Collapse potential states

2. **Binary System Interface** (`binary/`)
   - [ ] `system_interface.py` - Interface with binary systems
   - [ ] `state_mapper.py` - Map symbolic states to binary states

---

## Original Architecture Documentation

```
# Gola - AI-Powered Data Processing & Dataset Creation System

## Project Overview

Gola is a comprehensive CLI-first, MCP-ready system that ingests data from multiple sources (files, books, academic PDFs, web) and automatically builds clean, validated datasets with vectorization. It eliminates manual data collection and cleaning through intelligent automation and multi-model collaboration.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Ingestion      │    │   Processing    │
│                 │    │  Pipeline       │    │   Pipeline      │
│ • Files/Folders │───▶│                 │───▶│                 │
│ • Web URLs      │    │ • LLM-Ready     │    │ • Chunking      │
│ • APIs          │    │   Probe         │    │ • Deduplication │
│ • Databases     │    │ • Crawl4AI      │    │ • Normalization │
└─────────────────┘    │ • Document      │    │ • Vector        │
                       │   Parsing       │    │   Processing    │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Reasoner      │    │   Builders      │
                       │   (Orchestrator)│    │                 │
                       │                 │    │ • Summary       │
                       │ • PlanSpec      │    │ • Entities      │
                       │ • GPU Monitor   │    │ • Triples       │
                       │ • Router        │    │ • QA Pairs      │
                       │ • Validator     │    │ • Domain Tasks  │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Validation    │    │   Export &      │
                       │   Pipeline      │    │   Indexing      │
                       │                 │    │                 │
                       │ • Schema Check  │    │ • JSONL/CSV     │
                       │ • Cross-Validate│    │ • Parquet       │
                       │ • Quality Gates │    │ • Qdrant Index  │
                       │ • Adjudication  │    │ • Provenance    │
                       └─────────────────┘    └─────────────────┘
```

### Core Components

1. **CLI Interface** (`cli/`): Main entry point for user interactions
2. **MCP Server** (`mcp/`): Model Context Protocol integration
3. **Reasoner** (`reasoner/`): AI-driven planning and orchestration
4. **Pipeline** (`pipeline/`): Core data processing components
5. **Schemas** (`schemas/`): Pydantic models and validation
6. **Configs** (`configs/`): YAML configuration files

## Data Flow

### 1. Planning Phase
- **Reasoner** creates PlanSpec based on source and requirements
- **Scouting** probes for LLM-ready assets and estimates costs
- **Risk Assessment** evaluates duplication, PII, and compliance

### 2. Ingestion Phase
- **File Watching** monitors folders for new/changed files
- **LLM-Ready Probe** checks for `/llms.txt`, `.md variants`, plugin manifests
- **Crawl4AI Integration** scrapes web content when needed
- **Document Parsing** uses unstructured for PDF/HTML/DOCX processing

### 3. Processing Phase
- **Content-Defined Chunking** creates stable paragraph boundaries
- **Deduplication** uses SimHash/MinHash + LSH + vector similarity
- **Normalization** fixes encoding, hyphenation, and layout issues

### 4. Building Phase
- **GPU-Aware Routing** selects optimal models based on load and task
- **Multi-Model Collaboration** with cross-validation and evidence spans
- **Schema Validation** ensures type-safe outputs with Pydantic

### 5. Validation Phase
- **Quality Gates** check duplication ratios, coverage, and hallucination risk
- **Cross-Validation** multiple models verify each other's work
- **Adjudication** resolves conflicts and produces final outputs

### 6. Export Phase
- **Multiple Formats** JSONL, CSV, Parquet for different use cases
- **Vector Indexing** Qdrant database for semantic search
- **Provenance** complete audit trail with manifests

## Environment Variables & API Configuration

### Core AI Models
```bash
# Local Models (LM Studio)
LLAMA_MODEL_PATH=
LLAMA_MODEL_URL=
LLAMA_API_KEY=

# Embedding Model
QWEN_EMBEDDING_MODEL_PATH=
QWEN_EMBEDDING_MODEL_URL=
QWEN_API_KEY=

# Alternative Models
OPENAI_API_KEY=
ANTHROPIC_API_KEY_1=
COHERE_API_KEY=
DEEPSEEK_API_KEY_1=

# Grok AI (xAI)
GROK_API_KEY_0=
GROK_API_ENDPOINT=https://api.x.ai/v1/chat/completions
GROK_MODEL=grok-4-latest
XAI_SDK_ENABLED=true

# Local LM Studio
LM_STUDIO_API_LOCAL=http://192.168.56.1:11234

# Hugging Face
HF_TOKEN=

# Database
DB_NAME=ai_forge_db
DB_USER=ai_forge_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432

# Vector Store
VECTOR_DB_PATH=./vector_store/embeddings.db
VECTOR_DB_TYPE=sqlite
VECTOR_INDEX_CONFIG=./vector_store/index_config.yaml

# GitHub Integration
GITHUB_BASE_URL=https://github.com/AbramelinTheClown
```

## API Call Templates

### Grok AI Integration
```python
import requests
import os

def call_grok_api(messages, temperature=0, stream=False):
    """
    Call Grok AI API for chat completions
    """
    url = os.getenv('GROK_API_ENDPOINT', 'https://api.x.ai/v1/chat/completions')
    api_key = os.getenv('GROK_API_KEY_0')
    model = os.getenv('GROK_MODEL', 'grok-4-latest')
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    data = {
        'messages': messages,
        'model': model,
        'stream': stream,
        'temperature': temperature
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()
```

### Anthropic API Integration
```python
def call_anthropic_api(messages, model="claude-3-sonnet-20240229", max_tokens=4096):
    """
    Call Anthropic Claude API for chat completions
    """
    url = "https://api.anthropic.com/v1/messages"
    api_key = os.getenv('ANTHROPIC_API_KEY_1')
    
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
    }
    
    data = {
        'model': model,
        'max_tokens': max_tokens,
        'messages': messages
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()
```

### Anthropic Message Batches Optimization
```python
def create_anthropic_batch(requests_list):
    """
    Create a Message Batch for processing multiple requests efficiently
    """
    url = "https://api.anthropic.com/v1/messages/batches"
    api_key = os.getenv('ANTHROPIC_API_KEY_1')
    
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
    }
    
    data = {
        'requests': requests_list
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def get_batch_results(batch_id):
    """
    Retrieve results from a completed Message Batch
    """
    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
    api_key = os.getenv('ANTHROPIC_API_KEY_1')
    
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
    }
    
    response = requests.get(url, headers=headers)
    return response.json()
```

### Grok AI Structured Outputs
```python
def call_grok_structured_output(prompt, schema_class, temperature=0):
    """
    Call Grok AI with Structured Outputs for type-safe data extraction
    """
    from xai_sdk import Client
    from xai_sdk.chat import system, user
    
    client = Client(api_key=os.getenv("XAI_API_KEY"))
    chat = client.chat.create(model="grok-4")
    chat.append(system("Extract structured data according to the schema."))
    chat.append(user(prompt))
    
    response, parsed_data = chat.parse(schema_class)
    return parsed_data
```

### DeepSeek API Integration
```python
def call_deepseek_api(messages, model="deepseek-chat", temperature=0):
    """
    Call DeepSeek API for chat completions
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    api_key = os.getenv('DEEPSEEK_API_KEY_1')
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    data = {
        'model': model,
        'messages': messages,
        'temperature': temperature
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()
```

## Recent Updates

### 2024-12-19: CLI Commands Implementation
- ✅ Implemented comprehensive CLI command system with Typer integration
- ✅ Created plan, ingest, build, export, validate, crawl, mcp, and hub commands
- ✅ Added rich output formatting with tables and progress indicators
- ✅ Implemented status monitoring and validation for all operations
- ✅ Created demo script (`test_cli_demo.py`) showcasing full workflow
- ✅ Added comprehensive error handling and logging throughout CLI

### 2024-12-19: Symbolic Extraction System
- ✅ Implemented comprehensive symbolic extraction from Liber777revised JP2 files
- ✅ Created training datasets for LoRA fine-tuning on symbolic reasoning
- ✅ Added pattern recognition for geometric, numerical, and textual patterns
- ✅ Implemented correspondence extraction from multiple modalities

### 2024-12-19: Framework-First Architecture
- 🔄 Transitioning to pip-installable framework approach (`perspective-dcide`)
- 🔄 Implementing Emergent TaXonomy (ETX) as first-class framework component
- 🔄 Developing symbolic reasoning capabilities for future state prediction

### 2024-12-19: Future States & Symbolic Reasoning
- 🔄 Conceptual framework for representing "different future states of things"
- 🔄 System hashing and placeholder scaffolding for undefined code
- 🔄 Agent scaffolding using symbolic JSONL "flags of potential"
- 🔄 Recursive collapse of potentials for symbolic reasoning in binary systems

## Next Steps

### Immediate Priorities:
1. **Complete Framework Packaging**: Package as `perspective-dcide` pip installable framework
2. **ETX Implementation**: Complete Emergent TaXonomy system
3. **Web Scraping**: Integrate Crawl4AI for web content ingestion
4. **MCP Server**: Build MCP server for external LLM integration
5. **Symbolic State Extractor**: Implement future state prediction and symbolic reasoning

### Medium-term Goals:
1. **Hub Integration**: Complete cross-project knowledge sharing
2. **Advanced Validation**: Multi-model consensus and quality gates
3. **Export Pipeline**: Multiple format export with vector indexing
4. **Performance Optimization**: Parallel processing and caching improvements

### Long-term Vision:
1. **Symbolic Reasoning**: Full implementation of future state prediction
2. **Binary System Interface**: Direct interface with binary systems
3. **Agent Scaffolding**: Complete agent-driven process scaffolding
4. **Recursive Collapse**: Efficient collapse of all potential states

## CLI Usage Guide

### Quick Start
```bash
# Initialize the system
corpusctl init

# Check system status
corpusctl status

# Plan a processing operation
corpusctl plan create ./documents --mode general --budget 25 --verbose

# Ingest files
corpusctl ingest files ./documents --output ./data/ingested

# Build dataset
corpusctl build dataset ./data/ingested --tasks summary,entities

# Export results
corpusctl export dataset ./data/datasets/dataset.jsonl --formats jsonl,csv,parquet
```

### Advanced Usage
```bash
# Watch for new files and process automatically
corpusctl ingest files ./documents --watch --recursive

# Build with parallel processing
corpusctl build dataset ./data/ingested --tasks summary,entities,qa_pairs --parallel 8

# Export with vectorization
corpusctl export dataset ./data/datasets/dataset.jsonl --formats jsonl,csv,parquet --vectorize

# Validate dataset quality
corpusctl validate dataset ./data/datasets/dataset.jsonl --verbose

# Sync with knowledge hub
corpusctl hub sync --project my-project --content-type patterns,concepts
```

### Configuration
```bash
# Use custom configuration
corpusctl --config ./my_config.yaml plan create ./documents

# Enable verbose logging
corpusctl --verbose ingest files ./documents

# Enable debug mode
corpusctl --debug build dataset ./data/ingested
```

## System Requirements

### Dependencies
- Python 3.8+
- Typer (CLI framework)
- Rich (console output)
- Pydantic (data validation)
- PyYAML (configuration)
- Pandas (data processing)
- NumPy (numerical operations)

### Optional Dependencies
- PyTorch (GPU processing)
- Transformers (AI models)
- Unstructured (document parsing)
- Tesseract (OCR)
- Qdrant (vector database)
- PostgreSQL (database)

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize system
corpusctl init
corpusctl doctor
``` 