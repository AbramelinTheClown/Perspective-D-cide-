# Dev Vector DB Hub Integration Guide

## Overview

The Dev Vector DB Hub serves as a central knowledge repository for all your projects, providing cross-project memory and insights. This guide explains how to integrate the hub with the Gola system for seamless knowledge sharing and discovery.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gola System   │    │   Hub Client    │    │   Dev Vector DB │
│                 │    │                 │    │   Hub           │
│ • Pipeline      │───▶│ • REST Client   │───▶│ • ChromaDB      │
│ • Datasets      │    │ • MCP Client    │    │ • all-MiniLM    │
│ • Insights      │    │ • Sync Adapter  │    │ • REST API      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Content Types │    │   Access Points │
                       │                 │    │                 │
                       │ • Patterns      │    │ • REST: 8003    │
                       │ • Concepts      │    │ • MCP: 9999     │
                       │ • Documentation │    │ • Swagger: /docs │
                       │ • Ideas         │    │                 │
                       │ • Code          │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## Setup

### 1. Start the Hub

```bash
# Start the Dev Vector DB Hub
python start_vectorization_system.py

# This starts:
# - REST API on port 8003
# - MCP server on port 9999
# - Swagger docs at http://localhost:8003/docs
```

### 2. Configure Gola

Update your `configs/hub.yaml`:

```yaml
hub:
  rest_host: localhost
  rest_port: 8003
  mcp_host: localhost
  mcp_port: 9999
  
  # Set your API key
  api_key: "your_hub_api_key"
  
  # Embedding configuration
  embedding_model: "all-MiniLM-L6-v2"
  embedding_dimension: 384
```

### 3. Register a Project

```bash
# Register your project with the hub
corpusctl hub register "my_project" --description "My data processing project"

# This creates a project ID and stores it in your config
```

## Usage Examples

### 1. Basic Workflow

```bash
# 1. Plan and process data
corpusctl plan --source ./library --mode fiction --budget 20
corpusctl ingest --source ./library --notes
corpusctl build --tasks summary,entities,qa --mode fiction

# 2. Validate and export
corpusctl validate --dataset books_v1
corpusctl export --dataset books_v1 --format jsonl,csv,parquet

# 3. Push insights to hub
corpusctl hub push books_v1 --project my_project

# 4. Search for patterns
corpusctl hub search "entity extraction patterns" --types pattern,concept
```

### 2. Cross-Project Knowledge Discovery

```bash
# Search for patterns across all projects
corpusctl hub search "successful model combinations" --types pattern

# Get project context
corpusctl hub context --project my_project

# Get development coordinates
corpusctl hub coordinates "how to handle large PDF files"
```

### 3. Manual Content Storage

```bash
# Store a pattern
corpusctl hub store \
  "Use Claude-3.7-sonnet for entity extraction tasks" \
  --type pattern \
  --metadata '{"task_type": "entity_extraction", "success_rate": 0.95}'

# Store a concept
corpusctl hub store \
  "Academic papers often contain structured abstracts" \
  --type concept \
  --metadata '{"domain": "academic", "frequency": "high"}'
```

## Content Types

### 1. Patterns
Algorithmic insights, successful model combinations, and proven approaches.

**Examples:**
- "Model combination anthropic/claude-3.7-sonnet works well for entity extraction"
- "Use FastCDC with 1000-byte chunks for academic papers"
- "Anthropic Message Batches reduce costs by 50% for large datasets"

### 2. Concepts
Design decisions, architectural patterns, and domain knowledge.

**Examples:**
- "Entity type 'Person' appears frequently in fiction datasets"
- "Academic papers often contain structured abstracts"
- "Technical documents benefit from table extraction"

### 3. Documentation
Dataset manifests, playbooks, and project documentation.

**Examples:**
- "Dataset fiction_v1 processed with 150 runs, quality score: 0.92"
- "Best practices for handling PII in legal documents"
- "Configuration template for technical document processing"

### 4. Ideas
New hypotheses, ablation takeaways, and follow-up experiments.

**Examples:**
- "Try using Grok for long-context summarization"
- "Investigate cross-lingual entity extraction"
- "Explore multimodal table extraction from PDFs"

### 5. Code
Code snippets, implementations, and reusable components.

**Examples:**
- "Python function for FastCDC chunking"
- "Pydantic schema for entity validation"
- "Qdrant collection configuration"

## Sync Policies

### Privacy Settings

```yaml
sync_policies:
  # Anonymize PII before pushing
  anonymize_pii: true
  
  # Remove sensitive metadata
  remove_sensitive_metadata: true
  
  # Enable cross-project search (use with caution)
  cross_project_search: false
```

### Quality Thresholds

```yaml
sync_policies:
  # Minimum confidence for insights
  min_confidence: 0.7
  
  # Minimum coverage for patterns
  min_coverage: 0.8
  
  # Maximum duplication ratio
  max_duplication: 0.1
```

### Content Filtering

```yaml
sync_policies:
  # What gets pushed to hub
  push_patterns: true
  push_concepts: true
  push_documentation: true
  push_ideas: true
  push_qa_samples: false  # Set to true for anonymized samples
```

## API Integration

### REST API Endpoints

```python
# Search for content
GET /search
POST /search
{
    "query": "entity extraction patterns",
    "embedding": [0.1, 0.2, ...],
    "limit": 10,
    "threshold": 0.7,
    "project_id": "my_project"
}

# Store content
POST /store
{
    "content": "Use Claude-3.7-sonnet for entity extraction",
    "content_type": "pattern",
    "metadata": {"task_type": "entity_extraction"},
    "embedding": [0.1, 0.2, ...]
}

# Get project context
GET /projects/{project_id}/context

# Get statistics
GET /stats
GET /stats/project/{project_id}
```

### MCP Tools

```python
# Available MCP tools
vector_db_query(query: str, content_types: List[str], limit: int)
vector_db_store(content: str, content_type: str, metadata: Dict)
development_coordinate(query: str)
get_project_context(project_id: str)
```

## Advanced Features

### 1. Automatic Insight Extraction

The system automatically extracts insights from your datasets:

**From Runs:**
- Successful model combinations
- Task-specific performance patterns
- Cost optimization strategies

**From Entities:**
- Common entity types
- Domain-specific patterns
- Frequency distributions

**From Triples:**
- Common predicates
- Relationship patterns
- Knowledge graph insights

**From QA Pairs:**
- Question type patterns
- Answer type distributions
- Domain-specific question patterns

### 2. Cross-Project Learning

```bash
# Search for patterns across all projects
corpusctl hub search "successful model combinations" --types pattern

# Get context for a new project
corpusctl hub context --project new_project

# Find similar datasets
corpusctl hub search "academic paper processing" --types documentation
```

### 3. Development Coordinates

```bash
# Get development guidance
corpusctl hub coordinates "how to handle large PDF files"

# Find relevant patterns
corpusctl hub search "PDF processing" --types pattern,concept

# Get project-specific insights
corpusctl hub context --project my_project
```

## Monitoring and Analytics

### 1. Hub Statistics

```bash
# Get global stats
corpusctl hub stats

# Get project-specific stats
corpusctl hub stats --project my_project
```

### 2. Sync Monitoring

```bash
# Check sync status
corpusctl hub search "sync status" --types documentation

# View sync summary
SELECT * FROM hub_sync_summary;
```

### 3. Quality Metrics

- **Coverage**: Percentage of insights with evidence spans
- **Confidence**: Average confidence scores
- **Duplication**: Ratio of duplicate insights
- **Relevance**: Search result relevance scores

## Best Practices

### 1. Content Organization

- Use descriptive content for patterns and concepts
- Include relevant metadata for better searchability
- Tag content with appropriate domains and technologies

### 2. Privacy and Security

- Always anonymize PII before pushing to hub
- Use project isolation for sensitive data
- Review content before cross-project sharing

### 3. Quality Assurance

- Set appropriate confidence thresholds
- Filter out low-quality insights
- Regularly review and clean up old content

### 4. Performance Optimization

- Use caching for frequently accessed content
- Batch sync operations for large datasets
- Monitor API usage and costs

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if hub is running
   curl http://localhost:8003/health
   
   # Verify configuration
   corpusctl hub stats
   ```

2. **Authentication Error**
   ```bash
   # Check API key
   echo $HUB_API_KEY
   
   # Re-register project
   corpusctl hub register "my_project"
   ```

3. **Sync Failures**
   ```bash
   # Check sync status
   SELECT * FROM hub_sync WHERE status = 'failed';
   
   # Retry failed syncs
   corpusctl hub push dataset_name --force
   ```

### Debug Mode

```bash
# Enable debug logging
corpusctl --debug hub search "test query"

# Check hub logs
tail -f logs/hub.log
```

## Integration with Other Tools

### 1. MCP Integration

The hub can be accessed by any MCP-compatible client:

```python
# Example MCP client usage
from mcp import Client

client = Client("localhost", 9999)
results = client.call("vector_db_query", {
    "query": "entity extraction patterns",
    "content_types": ["pattern", "concept"],
    "limit": 10
})
```

### 2. REST API Integration

```python
# Example REST client usage
import requests

response = requests.post("http://localhost:8003/search", json={
    "query": "entity extraction patterns",
    "limit": 10,
    "threshold": 0.7
})
```

### 3. Swagger Documentation

Access the interactive API documentation at:
```
http://localhost:8003/docs
```

This provides a complete reference for all available endpoints and their parameters.

## Future Enhancements

### Planned Features

1. **Advanced Analytics**
   - Trend analysis across projects
   - Performance correlation studies
   - Automated insight generation

2. **Collaborative Features**
   - Multi-user access control
   - Comment and discussion system
   - Version control for insights

3. **Integration Ecosystem**
   - GitHub integration for code insights
   - Jupyter notebook integration
   - CI/CD pipeline integration

4. **Advanced Search**
   - Semantic search improvements
   - Multi-modal search (text + code)
   - Contextual search with project history

This integration provides a powerful foundation for building a comprehensive knowledge management system that grows with your projects and helps you avoid repeating work across different initiatives. 