# Gola - AI-Powered Data Processing & Dataset Creation System

## Overview

Gola is a comprehensive CLI-first, MCP-ready system that ingests data from multiple sources (files, books, academic PDFs, web) and automatically builds clean, validated datasets with vectorization. It eliminates manual data collection and cleaning through intelligent automation and multi-model collaboration.

## Key Features

- **Self-Planning & Reasoning**: AI-driven orchestration with continuous monitoring and adaptation
- **Multi-Model Collaboration**: Models cross-check each other's work for quality assurance
- **GPU-Aware Routing**: LM Studio first, with intelligent cloud bursting
- **LLM-Ready Web Intelligence**: Probes for pre-made content before scraping
- **Quality Assurance**: Evidence spans, schema validation, and cross-validation
- **MCP Integration**: Exposes tools and resources for other agents

## Architecture

### System Components

```
[Data Sources] → [Ingestion Pipeline] → [Processing Pipeline] → [Output Generation]
     │                    │                       │                      │
     ▼                    ▼                       ▼                      ▼
[Files/Web/APIs] → [Parse & Normalize] → [Chunk & Dedup] → [Build Datasets]
     │                    │                       │                      │
     ▼                    ▼                       ▼                      ▼
[LLM-Ready Probe] → [Layout Analysis] → [Vector Processing] → [Export & Index]
```

### Core Modules

1. **CLI Interface** (`cli/`): Main command-line interface
2. **MCP Server** (`mcp/`): Model Context Protocol integration
3. **Reasoner** (`reasoner/`): AI-driven planning and orchestration
4. **Pipeline** (`pipeline/`): Core data processing components
5. **Schemas** (`schemas/`): Pydantic models and validation
6. **Configs** (`configs/`): YAML configuration files

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd gola

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys and settings
```

### Basic Usage

```bash
# Plan a data processing run
corpusctl plan --source ./library --mode fiction --budget 20

# Ingest and process data
corpusctl ingest --source ./library --notes

# Build datasets with GPU-aware routing
corpusctl build --tasks summary,entities,qa --mode fiction

# Validate and export
corpusctl validate --dataset books_v1
corpusctl export --dataset books_v1 --format jsonl,csv,parquet

# Serve MCP tools for other agents
corpusctl mcp serve --port 3323
```

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

## Configuration

### Environment Variables

```bash
# Core AI Models
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

### Project Configuration

```yaml
# configs/project.yaml
default_mode: general
router_policy: throughput
providers:
  lmstudio:
    base_url: http://127.0.0.1:1234/v1
  anthropic: { enabled: true }
  gemini: { enabled: true }
  deepseek: { enabled: true }
budgets:
  daily_usd: 20
```

## Modes & Domains

### General Mode
- **Tasks**: summary, entities, triples, qa_pairs, topics
- **Use Case**: General document processing and analysis

### Fiction Mode
- **Tasks**: characters, timeline, dialogue_acts, themes, style
- **Use Case**: Literary analysis and character mapping

### Technical Mode
- **Tasks**: equations, figures, tables, citations, definitions
- **Use Case**: Academic and technical document processing

### Legal Mode
- **Tasks**: sections, clauses, obligations, rights, terms
- **Use Case**: Legal document analysis and compliance

## Quality Assurance

### Evidence Spans
All outputs include character-level references back to source text:
```json
{
  "summary": "The document discusses...",
  "evidence_spans": [
    {"start": 18430, "end": 18495},
    {"start": 18610, "end": 18670}
  ]
}
```

### Schema Validation
Pydantic models ensure type-safe outputs:
```python
class SummaryOutput(BaseModel):
    summary_text: str
    keypoints: List[str]
    evidence_spans: List[Dict[str, int]]
    confidence: float = Field(ge=0, le=1)
```

### Cross-Validation
Multiple models verify each other's work:
- **Self-Consistency**: N-best sampling with voting
- **Critic**: Checklist-based validation
- **Adjudicator**: Conflict resolution and merging

## Performance & Monitoring

### GPU Monitoring
- **NVML Integration**: Real-time GPU telemetry
- **Load Balancing**: Automatic routing based on GPU utilization
- **Resource Management**: VRAM-aware batch sizing

### Cost Optimization
- **Provider Rotation**: Intelligent routing to minimize costs
- **Budget Controls**: Daily spending limits and alerts
- **Semantic Caching**: Avoid redundant API calls

### Quality Metrics
- **Duplication Ratio**: Track and minimize duplicate content
- **Coverage Score**: Ensure comprehensive processing
- **Hallucination Risk**: Detect claims without evidence spans

## Development

### Project Structure
```
gola/
├── cli/                    # Main CLI commands
├── mcp/                    # MCP server tools/resources
├── reasoner/               # Planning & orchestration
├── pipeline/
│   ├── ingest/            # File/web intake
│   ├── normalize/         # Text processing
│   ├── chunk/            # Content-defined chunking
│   ├── dedup/            # SimHash/MinHash + vector
│   ├── builders/          # Task-specific extractors
│   ├── validate/          # Quality checks
│   └── export/            # JSONL/CSV/Parquet
├── schemas/               # Pydantic models
├── configs/               # YAML configurations
├── data/                  # Processed data
└── indexes/               # Vector databases
```

### Adding New Tasks
1. Create Pydantic schema in `schemas/`
2. Implement builder in `pipeline/builders/`
3. Add validation rules in `pipeline/validate/`
4. Update mode configurations in `configs/modes/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[License information to be added]

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the examples in `examples/` 