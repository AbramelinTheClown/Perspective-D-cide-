# Perspective D<cide> Framework Documentation

## 🏗️ **Framework Overview**

Perspective D<cide> is a comprehensive AI-powered framework for dynamic content analysis, categorization, and knowledge organization. It provides a modular, extensible architecture for building intelligent systems that can understand, organize, and process content at scale.

### **Core Philosophy**
- **Emergent Intelligence**: Categories and taxonomies emerge from content analysis, not pre-defined schemas
- **Framework-First**: Designed as a core component that can be embedded in larger systems
- **Zero-Setup**: Minimal configuration required for basic functionality
- **Extensible**: Plugin architecture for custom behaviors and integrations
- **Production-Ready**: Built for scale with async processing, error handling, and monitoring

## 📦 **Installation & Integration**

### **Basic Installation**

```bash
# Core framework
pip install perspective-dcide

# With optional extras
pip install "perspective-dcide[full]"  # All dependencies
pip install "perspective-dcide[etx]"   # ETX components only
pip install "perspective-dcide[ml]"    # Machine learning extras
```

### **Integration into Larger Packages**

#### **1. As a Framework Component**

```python
# In your package's setup.py or pyproject.toml
[project]
name = "your-ai-package"
version = "1.0.0"
dependencies = [
    "perspective-dcide>=1.0.0",
    "numpy>=1.26",
    "pandas>=2.2",
    # ... other dependencies
]

[project.optional-dependencies]
etx = ["perspective-dcide[etx]"]
ml = ["perspective-dcide[ml]"]
full = ["perspective-dcide[full]"]
```

#### **2. Framework Initialization**

```python
# In your package's __init__.py
from perspective_dcide.core import initialize_framework
from perspective_dcide.core.config import Config

def initialize_your_package():
    """Initialize your package with Perspective D<cide> integration."""
    
    # Initialize framework
    config = Config(
        framework_name="your-package",
        enable_etx=True,
        enable_ml=True,
        storage_backend="sqlite",
        log_level="INFO"
    )
    
    initialize_framework(config)
    
    # Your package initialization code here
    pass
```

#### **3. Component Registration**

```python
# Register your components with the framework
from perspective_dcide.core.registry import ComponentRegistry
from perspective_dcide.etx import CategorizationPlugin

class YourCustomPlugin(CategorizationPlugin):
    """Your custom categorization logic."""
    
    def analyze_content(self, content: str, context: Dict[str, Any]) -> List[CategoryProposal]:
        # Your custom analysis
        return []
    
    def validate_categories(self, categories: List[CategoryProposal]) -> List[CategoryProposal]:
        # Your custom validation
        return categories

# Register with framework
registry = ComponentRegistry()
registry.register_plugin("your_custom", YourCustomPlugin())
```

## 🏛️ **Architecture**

### **Core Framework Structure**

```
perspective_dcide/
├── core/                          # Core framework components
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── logging.py                 # Logging system
│   ├── storage.py                 # Storage backends
│   ├── registry.py                # Component registry
│   ├── schemas.py                 # Core data models
│   └── utils.py                   # Utility functions
├── etx/                           # Emergent TaXonomy system
│   ├── __init__.py
│   ├── framework.py               # Main ETX framework
│   ├── builders.py                # Categorization builders
│   ├── engines.py                 # Embedding & clustering engines
│   ├── plugins.py                 # Plugin interfaces
│   └── storage.py                 # ETX-specific storage
├── ml/                            # Machine learning components
│   ├── __init__.py
│   ├── models.py                  # ML model interfaces
│   ├── training.py                # Training utilities
│   └── inference.py               # Inference engines
├── pipeline/                      # Processing pipeline
│   ├── __init__.py
│   ├── ingest.py                  # Data ingestion
│   ├── process.py                 # Data processing
│   ├── validate.py                # Validation
│   └── export.py                  # Data export
├── cli/                           # Command-line interface
│   ├── __init__.py
│   ├── main.py                    # Main CLI entry point
│   └── commands/                  # CLI commands
└── plugins/                       # Built-in plugins
    ├── __init__.py
    ├── categorization/            # Categorization plugins
    ├── analysis/                  # Analysis plugins
    └── export/                    # Export plugins
```

### **Component Architecture**

#### **1. Core Framework (`core/`)**

The core framework provides the foundation for all other components:

```python
from perspective_dcide.core import Config, initialize_framework
from perspective_dcide.core.storage import StorageBackend
from perspective_dcide.core.registry import ComponentRegistry

# Configuration
config = Config(
    framework_name="my-application",
    storage_backend="sqlite",
    log_level="INFO",
    enable_async=True,
    max_workers=4
)

# Initialize framework
initialize_framework(config)

# Access core components
storage = StorageBackend.get_instance()
registry = ComponentRegistry.get_instance()
```

#### **2. ETX System (`etx/`)**

The Emergent TaXonomy system provides dynamic categorization:

```python
from perspective_dcide.etx import ETXFramework, CategorizationBuilder
from perspective_dcide.etx.engines import FastEmbedEngine, MiniBatchKMeansEngine

# Create ETX framework
etx = ETXFramework(
    project="my_project",
    embedding_engine=FastEmbedEngine(),
    clustering_engine=MiniBatchKMeansEngine()
)

# Ingest and categorize content
etx.ingest("./data.jsonl")
etx.discover()
results = etx.export(as_df=True)
```

#### **3. Machine Learning (`ml/`)**

ML components for advanced analysis:

```python
from perspective_dcide.ml import ModelManager, TrainingPipeline
from perspective_dcide.ml.models import TextClassificationModel

# Model management
model_manager = ModelManager()
model = TextClassificationModel(
    model_type="transformer",
    pretrained_model="bert-base-uncased"
)

# Training pipeline
pipeline = TrainingPipeline(model)
pipeline.train(training_data, validation_data)
```

## 🔧 **Configuration System**

### **Framework Configuration**

```python
from perspective_dcide.core.config import Config

config = Config(
    # Basic settings
    framework_name="my-application",
    version="1.0.0",
    
    # Storage configuration
    storage_backend="sqlite",  # sqlite, postgres, redis
    storage_path="~/.perspective_dcide",
    
    # Logging configuration
    log_level="INFO",
    log_file="perspective_dcide.log",
    
    # Processing configuration
    enable_async=True,
    max_workers=4,
    batch_size=1000,
    
    # Component enablement
    enable_etx=True,
    enable_ml=True,
    enable_pipeline=True,
    
    # ETX-specific settings
    etx=Config(
        embedding_model="bge-small-en",
        clustering_type="minibatch_kmeans",
        consensus_threshold=0.6,
        min_confidence=0.7
    ),
    
    # ML-specific settings
    ml=Config(
        model_cache_dir="~/.cache/perspective_dcide/models",
        enable_gpu=True,
        precision="float16"
    )
)
```

### **Environment Variables**

```bash
# Framework configuration
export PERSPECTIVE_DCIDE_FRAMEWORK_NAME="my-app"
export PERSPECTIVE_DCIDE_STORAGE_BACKEND="sqlite"
export PERSPECTIVE_DCIDE_LOG_LEVEL="INFO"

# ETX configuration
export PERSPECTIVE_DCIDE_ETX_EMBEDDING_MODEL="bge-small-en"
export PERSPECTIVE_DCIDE_ETX_CLUSTERING_TYPE="minibatch_kmeans"

# ML configuration
export PERSPECTIVE_DCIDE_ML_ENABLE_GPU="true"
export PERSPECTIVE_DCIDE_ML_PRECISION="float16"
```

## 🚀 **Usage Examples**

### **1. Basic Framework Usage**

```python
from perspective_dcide.core import initialize_framework, Config
from perspective_dcide.etx import ETXFramework

# Initialize framework
config = Config(
    framework_name="content-analyzer",
    enable_etx=True,
    storage_backend="sqlite"
)
initialize_framework(config)

# Use ETX for content categorization
etx = ETXFramework("my_project")
etx.ingest("./content.jsonl")
etx.discover()

# Export results
results = etx.export(as_df=True)
print(f"Discovered {len(results)} content items")
```

### **2. Advanced Pipeline Usage**

```python
from perspective_dcide.pipeline import ProcessingPipeline
from perspective_dcide.etx import CategorizationBuilder
from perspective_dcide.ml import TextAnalysisModel

# Create processing pipeline
pipeline = ProcessingPipeline()

# Add components
pipeline.add_builder(CategorizationBuilder())
pipeline.add_model(TextAnalysisModel())

# Process data
results = pipeline.process(
    source="./data/",
    output="./results/",
    config={
        "categorization": {"min_confidence": 0.8},
        "analysis": {"model_type": "sentiment"}
    }
)
```

### **3. Plugin Development**

```python
from perspective_dcide.etx import CategorizationPlugin
from perspective_dcide.core.registry import ComponentRegistry

class CustomCategorizationPlugin(CategorizationPlugin):
    """Custom categorization logic for domain-specific content."""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.keywords = self._load_domain_keywords(domain)
    
    def analyze_content(self, content: str, context: Dict[str, Any]) -> List[CategoryProposal]:
        """Analyze content using domain-specific rules."""
        categories = []
        
        # Domain-specific analysis
        for keyword, category_info in self.keywords.items():
            if keyword.lower() in content.lower():
                categories.append(CategoryProposal(
                    category_name=category_info["name"],
                    confidence=category_info["confidence"],
                    reasoning=f"Contains domain keyword: {keyword}",
                    agent_id=f"domain_{self.domain}",
                    keywords=[keyword],
                    description=category_info["description"]
                ))
        
        return categories
    
    def _load_domain_keywords(self, domain: str) -> Dict[str, Dict]:
        """Load domain-specific keywords and categories."""
        # Implementation for loading domain keywords
        return {}

# Register plugin
registry = ComponentRegistry.get_instance()
registry.register_plugin("domain_specific", CustomCategorizationPlugin("finance"))
```

### **4. Integration with External Systems**

```python
from perspective_dcide.core import Config, initialize_framework
from perspective_dcide.etx import ETXFramework
from perspective_dcide.core.storage import PostgresBackend

# Initialize with external database
config = Config(
    framework_name="enterprise-analyzer",
    storage_backend="postgres",
    storage_config={
        "host": "localhost",
        "port": 5432,
        "database": "content_analysis",
        "username": "user",
        "password": "password"
    }
)
initialize_framework(config)

# Use ETX with external storage
etx = ETXFramework("enterprise_project")
etx.ingest("./enterprise_data.jsonl")
etx.discover()

# Export to external system
results = etx.export(as_df=True)
results.to_sql("categorized_content", con=engine, if_exists="append")
```

## 🔌 **Plugin System**

### **Plugin Architecture**

The framework uses a comprehensive plugin system for extensibility:

```python
from perspective_dcide.core.plugins import PluginInterface, PluginManager
from abc import abstractmethod
from typing import Dict, Any, List

class AnalysisPlugin(PluginInterface):
    """Base interface for analysis plugins."""
    
    @abstractmethod
    def analyze(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content and return results."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities."""
        pass

class SentimentAnalysisPlugin(AnalysisPlugin):
    """Sentiment analysis plugin."""
    
    def analyze(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implement sentiment analysis
        return {"sentiment": "positive", "confidence": 0.85}
    
    def get_capabilities(self) -> List[str]:
        return ["sentiment_analysis", "confidence_scoring"]

# Plugin management
plugin_manager = PluginManager()
plugin_manager.register_plugin("sentiment", SentimentAnalysisPlugin())
```

### **Built-in Plugins**

The framework comes with several built-in plugins:

```python
from perspective_dcide.plugins import (
    TopicAnalysisPlugin,
    EntityExtractionPlugin,
    SentimentAnalysisPlugin,
    LanguageDetectionPlugin,
    ContentClassificationPlugin
)

# Use built-in plugins
plugins = [
    TopicAnalysisPlugin(),
    EntityExtractionPlugin(),
    SentimentAnalysisPlugin(),
    LanguageDetectionPlugin(),
    ContentClassificationPlugin()
]

for plugin in plugins:
    plugin_manager.register_plugin(plugin.name, plugin)
```

## 📊 **Data Models & Schemas**

### **Core Data Models**

```python
from perspective_dcide.core.schemas import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class ContentItem(BaseModel):
    """Represents a piece of content."""
    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., description="Content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class CategoryProposal(BaseModel):
    """A proposed category for content."""
    category_name: str = Field(..., description="Category name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Reasoning for this category")
    agent_id: str = Field(..., description="Agent that proposed this category")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    description: str = Field("", description="Category description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AnalysisResult(BaseModel):
    """Result from content analysis."""
    content_id: str = Field(..., description="Content identifier")
    analysis_type: str = Field(..., description="Type of analysis")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

### **Schema Validation**

```python
from perspective_dcide.core.schemas import validate_schema
from pydantic import ValidationError

try:
    content_item = ContentItem(
        id="doc_001",
        content="Sample content for analysis",
        metadata={"source": "file", "language": "en"}
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

## ⚡ **Async Processing**

### **Async Framework Usage**

```python
import asyncio
from perspective_dcide.core import AsyncFramework
from perspective_dcide.etx import AsyncETXFramework

async def process_content():
    # Initialize async framework
    framework = AsyncFramework()
    await framework.initialize()
    
    # Use async ETX
    etx = AsyncETXFramework("async_project")
    
    # Async ingestion
    await etx.ingest_async("./large_dataset.jsonl")
    
    # Async discovery
    await etx.discover_async()
    
    # Async export
    results = await etx.export_async(as_df=True)
    
    return results

# Run async processing
results = asyncio.run(process_content())
```

### **Batch Processing**

```python
from perspective_dcide.pipeline import BatchProcessor

# Configure batch processing
processor = BatchProcessor(
    batch_size=1000,
    max_workers=4,
    enable_async=True
)

# Process large datasets
results = processor.process_batch(
    source="./large_dataset/",
    output="./processed_results/",
    config={
        "categorization": {"min_confidence": 0.7},
        "analysis": {"enable_sentiment": True}
    }
)
```

## 🧪 **Testing & Development**

### **Framework Testing**

```python
import pytest
from perspective_dcide.core import Config, initialize_framework
from perspective_dcide.etx import ETXFramework

@pytest.fixture
def framework():
    """Initialize framework for testing."""
    config = Config(
        framework_name="test-framework",
        storage_backend="sqlite",
        storage_path=":memory:",  # In-memory database for testing
        log_level="DEBUG"
    )
    initialize_framework(config)
    return config

@pytest.fixture
def etx_framework(framework):
    """Create ETX framework for testing."""
    return ETXFramework("test_project")

def test_content_categorization(etx_framework):
    """Test content categorization functionality."""
    # Test data
    test_content = [
        {"id": "1", "text": "Apple stock rises 5% after earnings report"},
        {"id": "2", "text": "New iPhone features announced at WWDC"},
        {"id": "3", "text": "Tesla reports record vehicle deliveries"}
    ]
    
    # Process test content
    for item in test_content:
        etx_framework._process_content_item(item, "text")
    
    # Run discovery
    etx_framework.discover()
    
    # Verify results
    results = etx_framework.export(as_df=True)
    assert len(results) > 0
    assert "cluster" in results.columns
```

### **Plugin Testing**

```python
import pytest
from perspective_dcide.core.plugins import PluginManager
from your_plugin import CustomAnalysisPlugin

@pytest.fixture
def plugin_manager():
    """Create plugin manager for testing."""
    return PluginManager()

def test_custom_plugin(plugin_manager):
    """Test custom plugin functionality."""
    plugin = CustomAnalysisPlugin()
    plugin_manager.register_plugin("custom", plugin)
    
    # Test plugin capabilities
    capabilities = plugin.get_capabilities()
    assert "custom_analysis" in capabilities
    
    # Test plugin analysis
    result = plugin.analyze("Test content", {})
    assert "analysis_result" in result
```

## 📈 **Performance & Scaling**

### **Performance Optimization**

```python
from perspective_dcide.core import Config
from perspective_dcide.etx import ETXFramework

# High-performance configuration
config = Config(
    framework_name="high-performance",
    enable_async=True,
    max_workers=8,
    batch_size=5000,
    storage_backend="postgres",
    storage_config={
        "pool_size": 20,
        "max_overflow": 30
    },
    etx=Config(
        embedding_model="bge-large-en",  # Higher quality embeddings
        clustering_type="hdbscan",       # Better clustering
        batch_processing=True,
        enable_caching=True,
        cache_size=10000
    )
)

# Use optimized framework
etx = ETXFramework("performance_project", config=config)
```

### **Distributed Processing**

```python
from perspective_dcide.distributed import DistributedFramework
from perspective_dcide.core import Config

# Distributed configuration
config = Config(
    framework_name="distributed",
    enable_distributed=True,
    distributed_config={
        "backend": "ray",
        "num_workers": 4,
        "memory_limit": "4GB"
    }
)

# Initialize distributed framework
framework = DistributedFramework(config)
framework.initialize()

# Process distributed data
results = framework.process_distributed(
    source="s3://bucket/data/",
    output="s3://bucket/results/",
    partitions=100
)
```

## 🔒 **Security & Privacy**

### **Data Privacy**

```python
from perspective_dcide.core import Config
from perspective_dcide.security import PrivacyManager

# Privacy-aware configuration
config = Config(
    framework_name="privacy-aware",
    enable_privacy=True,
    privacy_config={
        "pii_detection": True,
        "data_anonymization": True,
        "encryption": True,
        "retention_policy": "30_days"
    }
)

# Initialize privacy manager
privacy_manager = PrivacyManager(config)

# Process with privacy protection
protected_results = privacy_manager.process_with_privacy(
    content="Sensitive content with PII",
    privacy_level="high"
)
```

### **Access Control**

```python
from perspective_dcide.security import AccessControl

# Configure access control
access_control = AccessControl(
    authentication_backend="jwt",
    authorization_backend="rbac",
    audit_logging=True
)

# Secure framework access
secure_framework = access_control.secure_framework(
    framework=etx_framework,
    user_id="user_123",
    permissions=["read", "write", "analyze"]
)
```

## 📚 **API Reference**

### **Core Framework API**

```python
# Framework initialization
initialize_framework(config: Config) -> None

# Component registry
ComponentRegistry.register_component(name: str, component: Any) -> None
ComponentRegistry.get_component(name: str) -> Any

# Storage backends
StorageBackend.get_instance() -> StorageBackend
StorageBackend.store(key: str, value: Any) -> None
StorageBackend.retrieve(key: str) -> Any
```

### **ETX API**

```python
# ETX Framework
ETXFramework(project: str, config: Optional[Config] = None) -> ETXFramework
ETXFramework.ingest(source_path: str, modality: str = "text") -> None
ETXFramework.discover() -> None
ETXFramework.export(topic: str = "any", as_df: bool = True) -> pd.DataFrame

# Categorization Builder
CategorizationBuilder(config: Optional[BuilderConfig] = None) -> CategorizationBuilder
CategorizationBuilder.process_chunk(chunk: ChunkMetadata) -> BuilderResult
```

### **Plugin API**

```python
# Plugin interface
PluginInterface.initialize() -> None
PluginInterface.process(data: Any) -> Any
PluginInterface.cleanup() -> None

# Plugin management
PluginManager.register_plugin(name: str, plugin: PluginInterface) -> None
PluginManager.get_plugin(name: str) -> PluginInterface
PluginManager.list_plugins() -> List[str]
```

## 🚀 **Deployment & Production**

### **Docker Deployment**

```dockerfile
# Dockerfile for Perspective D<cide> application
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Perspective D<cide>
RUN pip install perspective-dcide[full]

# Copy application code
COPY . /app
WORKDIR /app

# Initialize framework
RUN python -c "from perspective_dcide.core import initialize_framework; initialize_framework()"

# Run application
CMD ["python", "app.py"]
```

### **Kubernetes Deployment**

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: perspective-dcide-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: perspective-dcide
  template:
    metadata:
      labels:
        app: perspective-dcide
    spec:
      containers:
      - name: app
        image: your-registry/perspective-dcide:latest
        ports:
        - containerPort: 8000
        env:
        - name: PERSPECTIVE_DCIDE_STORAGE_BACKEND
          value: "postgres"
        - name: PERSPECTIVE_DCIDE_DB_HOST
          value: "postgres-service"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### **Monitoring & Observability**

```python
from perspective_dcide.monitoring import MetricsCollector, HealthChecker

# Initialize monitoring
metrics = MetricsCollector()
health = HealthChecker()

# Monitor framework health
health.register_check("etx_processing", etx_framework.health_check)
health.register_check("storage_connection", storage.health_check)

# Collect metrics
metrics.record_metric("content_processed", len(processed_content))
metrics.record_metric("categories_discovered", len(discovered_categories))
metrics.record_metric("processing_time", processing_duration)

# Export metrics
metrics.export_to_prometheus()
```

## 🔄 **Migration & Upgrades**

### **Framework Version Migration**

```python
from perspective_dcide.migrations import MigrationManager

# Initialize migration manager
migration_manager = MigrationManager()

# Check for available migrations
available_migrations = migration_manager.get_available_migrations()

# Run migrations
for migration in available_migrations:
    migration_manager.run_migration(migration)
```

### **Data Migration**

```python
from perspective_dcide.migrations import DataMigrator

# Initialize data migrator
migrator = DataMigrator()

# Migrate data between versions
migrator.migrate_data(
    source_version="1.0.0",
    target_version="2.0.0",
    source_path="./old_data/",
    target_path="./new_data/"
)
```

## 📖 **Best Practices**

### **1. Framework Initialization**

```python
# Always initialize framework at application startup
from perspective_dcide.core import initialize_framework, Config

def main():
    config = Config(
        framework_name="my-application",
        enable_etx=True,
        storage_backend="sqlite"
    )
    initialize_framework(config)
    
    # Your application code here
    pass

if __name__ == "__main__":
    main()
```

### **2. Error Handling**

```python
from perspective_dcide.core.exceptions import FrameworkError, ProcessingError

try:
    etx = ETXFramework("my_project")
    etx.ingest("./data.jsonl")
    etx.discover()
except ProcessingError as e:
    logger.error(f"Processing error: {e}")
    # Handle processing errors
except FrameworkError as e:
    logger.error(f"Framework error: {e}")
    # Handle framework errors
```

### **3. Resource Management**

```python
from contextlib import contextmanager

@contextmanager
def framework_session():
    """Context manager for framework sessions."""
    try:
        # Initialize framework
        initialize_framework(config)
        yield
    finally:
        # Cleanup resources
        cleanup_framework()

# Use context manager
with framework_session():
    etx = ETXFramework("my_project")
    etx.ingest("./data.jsonl")
    etx.discover()
```

### **4. Configuration Management**

```python
import os
from perspective_dcide.core.config import Config

def load_config():
    """Load configuration from environment and files."""
    config = Config()
    
    # Load from environment variables
    config.framework_name = os.getenv("PERSPECTIVE_DCIDE_FRAMEWORK_NAME", "default")
    config.storage_backend = os.getenv("PERSPECTIVE_DCIDE_STORAGE_BACKEND", "sqlite")
    
    # Load from configuration file
    config_file = os.getenv("PERSPECTIVE_DCIDE_CONFIG_FILE")
    if config_file:
        config.load_from_file(config_file)
    
    return config
```

## 🎯 **Quick Start Guide**

### **1. Install the Framework**

```bash
pip install perspective-dcide[etx]
```

### **2. Create a Simple Application**

```python
# app.py
from perspective_dcide.core import initialize_framework, Config
from perspective_dcide.etx import ETXFramework

def main():
    # Initialize framework
    config = Config(
        framework_name="my-first-app",
        enable_etx=True,
        storage_backend="sqlite"
    )
    initialize_framework(config)
    
    # Create ETX framework
    etx = ETXFramework("demo_project")
    
    # Create sample data
    sample_data = [
        {"id": "1", "text": "Apple stock rises 5% after earnings report"},
        {"id": "2", "text": "New iPhone features announced at WWDC"},
        {"id": "3", "text": "Tesla reports record vehicle deliveries"},
        {"id": "4", "text": "Learn how to plant a vegetable garden"},
        {"id": "5", "text": "Cooking with cast iron skillets improves flavor"}
    ]
    
    # Save sample data
    import json
    with open("sample_data.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    # Process data
    etx.ingest("./sample_data.jsonl")
    etx.discover()
    
    # Export results
    results = etx.export(as_df=True)
    print("Categorization Results:")
    print(results)
    
    return results

if __name__ == "__main__":
    main()
```

### **3. Run the Application**

```bash
python app.py
```

## 🤝 **Contributing**

### **Development Setup**

```bash
# Clone the repository
git clone https://github.com/your-org/perspective-dcide.git
cd perspective-dcide

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
```

### **Code Style**

The framework follows these coding standards:
- **Python**: PEP 8 with Black formatting
- **Type Hints**: Required for all public APIs
- **Documentation**: Google-style docstrings
- **Testing**: pytest with 90%+ coverage requirement

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

### **Getting Help**

- **Documentation**: [https://perspective-dcide.readthedocs.io](https://perspective-dcide.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/perspective-dcide/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/perspective-dcide/discussions)
- **Email**: support@perspective-dcide.com

### **Community**

- **Slack**: [Join our Slack workspace](https://perspective-dcide.slack.com)
- **Discord**: [Join our Discord server](https://discord.gg/perspective-dcide)
- **Twitter**: [@PerspectiveDCide](https://twitter.com/PerspectiveDCide)

---

**Perspective D<cide>** - Building intelligent systems that understand and organize content at scale. 

## 🧹 **GPT-2 Integration & Cleanup System**

```python
#!/usr/bin/env python3
"""
GPT-2 Integration & Cleanup for Liber777revised Analysis
Using Perspective D<cide> Framework

This script:
1. Cleans out unnecessary GPT-2 files
2. Integrates GPT-2 with the dynamic analysis system
3. Creates a streamlined workflow for JP2 analysis
4. Removes garbage files and organizes the workspace
"""

import os
import shutil
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkspaceCleaner:
    """Cleans and organizes the workspace."""
    
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = Path(workspace_path)
        self.backup_path = self.workspace_path / "backup_before_cleanup"
        
        # Files to keep (essential)
        self.essential_files = {
            # Core analysis files
            "liber777_pdec_analysis.py",
            "run_liber777_analysis.py",
            "PDEC_ETX_FRAMEWORK.md",
            "requirements.txt",
            "README.md",
            "NOTES.md",
            
            # Analysis results
            "liber777_analysis_results_full/",
            "liber777_analysis.db",
            
            # GPT-2 essential files
            "gpt-2/src/",
            "gpt-2/download_model.py",
            "gpt-2/requirements.txt",
            "gpt-2/README.md",
            "gpt-2/LICENSE",
            
            # Framework files
            "test_core_components_simple.py",
            "pipeline/",
            "schemas/",
            "cli/",
            "configs/",
            "db/",
            "hub/",
            
            # Documentation
            "ARCHITECTURE.md",
            "DATA_FLOW.md",
            "HUB_INTEGRATION.md",
            "USAGE_GUIDE.md",
            "SYMPROJ_README.md",
            "SYMPROJ_SUMMARY.md",
            "LIBER777_ANALYSIS_SUMMARY.md"
        }
        
        # Files to remove (garbage)
        self.garbage_patterns = [
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
            "Thumbs.db",
            "*.log",
            "*.tmp",
            "*.bak",
            "*.swp",
            "*.swo",
            "*~",
            ".vscode/",
            ".idea/",
            "node_modules/",
            ".git/",
            "*.egg-info/",
            "dist/",
            "build/",
            "*.tar.gz",
            "*.zip"
        ]
        
        # GPT-2 files to remove (unnecessary)
        self.gpt2_garbage = [
            "gpt-2/.git/",
            "gpt-2/Dockerfile.cpu",
            "gpt-2/Dockerfile.gpu", 
            "gpt-2/DEVELOPERS.md",
            "gpt-2/CONTRIBUTORS.md",
            "gpt-2/.gitattributes",
            "gpt-2/.gitignore",
            "gpt-2/domains.txt",
            "gpt-2/model_card.md"
        ]
    
    def create_backup(self):
        """Create backup before cleanup."""
        logger.info("Creating backup before cleanup...")
        
        if self.backup_path.exists():
            shutil.rmtree(self.backup_path)
        
        self.backup_path.mkdir(exist_ok=True)
        
        # Copy all files to backup
        for item in self.workspace_path.iterdir():
            if item.name != "backup_before_cleanup":
                if item.is_file():
                    shutil.copy2(item, self.backup_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, self.backup_path / item.name)
        
        logger.info(f"Backup created at: {self.backup_path}")
    
    def clean_gpt2(self):
        """Clean unnecessary GPT-2 files."""
        logger.info("Cleaning GPT-2 directory...")
        
        gpt2_path = self.workspace_path / "gpt-2"
        if not gpt2_path.exists():
            logger.warning("GPT-2 directory not found")
            return
        
        # Remove garbage files
        for garbage_item in self.gpt2_garbage:
            item_path = gpt2_path / garbage_item.replace("gpt-2/", "")
            if item_path.exists():
                if item_path.is_file():
                    item_path.unlink()
                    logger.info(f"Removed: {item_path}")
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
                    logger.info(f"Removed: {item_path}")
        
        # Keep only essential GPT-2 files
        essential_gpt2_files = [
            "src/",
            "download_model.py", 
            "requirements.txt",
            "README.md",
            "LICENSE"
        ]
        
        for item in gpt2_path.iterdir():
            if item.name not in essential_gpt2_files and not item.name.startswith("models"):
                if item.is_file():
                    item.unlink()
                    logger.info(f"Removed GPT-2 file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    logger.info(f"Removed GPT-2 directory: {item}")
    
    def clean_workspace(self):
        """Clean the entire workspace."""
        logger.info("Cleaning workspace...")
        
        # Remove garbage patterns
        for pattern in self.garbage_patterns:
            if pattern.endswith("/"):
                # Directory pattern
                for item in self.workspace_path.rglob(pattern[:-1]):
                    if item.is_dir():
                        shutil.rmtree(item)
                        logger.info(f"Removed directory: {item}")
            else:
                # File pattern
                for item in self.workspace_path.rglob(pattern):
                    if item.is_file():
                        item.unlink()
                        logger.info(f"Removed file: {item}")
        
        # Remove duplicate analysis files
        duplicate_files = [
            "liber777_analysis_results/",  # Keep only liber777_analysis_results_full/
            "liber777_comprehensive_analysis/",
            "test_jp2_output/",
            "gola_js_analysis.json",
            "gola_project_analysis.json",
            "simple_python_analysis.json",  # Too large, keep only essential
            "advanced_analysis.json",
            "combined_analysis.json",
            "combined_project_analysis.json"
        ]
        
        for duplicate in duplicate_files:
            item_path = self.workspace_path / duplicate
            if item_path.exists():
                if item_path.is_file():
                    item_path.unlink()
                    logger.info(f"Removed duplicate: {item_path}")
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
                    logger.info(f"Removed duplicate directory: {item_path}")
    
    def organize_workspace(self):
        """Organize the workspace into logical directories."""
        logger.info("Organizing workspace...")
        
        # Create organized directory structure
        directories = {
            "analysis": ["liber777_analysis_results_full/"],
            "models": ["gpt-2/"],
            "framework": ["pipeline/", "schemas/", "cli/", "configs/", "db/", "hub/"],
            "docs": ["*.md"],
            "scripts": ["*.py"],
            "data": ["*.db", "*.json", "*.csv"]
        }
        
        for dir_name, patterns in directories.items():
            dir_path = self.workspace_path / dir_name
            dir_path.mkdir(exist_ok=True)
            
            for pattern in patterns:
                if pattern.endswith("/"):
                    # Directory pattern
                    pattern_name = pattern[:-1]
                    for item in self.workspace_path.glob(pattern_name):
                        if item.is_dir() and item.name != dir_name:
                            target = dir_path / item.name
                            if not target.exists():
                                shutil.move(str(item), str(target))
                                logger.info(f"Moved {item} to {target}")
                else:
                    # File pattern
                    for item in self.workspace_path.glob(pattern):
                        if item.is_file():
                            target = dir_path / item.name
                            if not target.exists():
                                shutil.move(str(item), str(target))
                                logger.info(f"Moved {item} to {target}")

class GPT2Integrator:
    """Integrates GPT-2 with the dynamic analysis system."""
    
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = Path(workspace_path)
        self.gpt2_path = self.workspace_path / "models" / "gpt-2"
        
    def create_integration_script(self):
        """Create script to integrate GPT-2 with Liber777revised analysis."""
        
        integration_script = '''#!/usr/bin/env python3
"""
GPT-2 Integration for Liber777revised Analysis
Using Perspective D<cide> Framework

This script integrates GPT-2 text generation with the dynamic Liber777revised analysis system.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add GPT-2 to path
sys.path.append(str(Path(__file__).parent / "models" / "gpt-2" / "src"))

try:
    import model, sample, encoder
    GPT2_AVAILABLE = True
except ImportError as e:
    print(f"GPT-2 not available: {e}")
    GPT2_AVAILABLE = False

class GPT2TextGenerator:
    """GPT-2 text generation for Liber777revised analysis."""
    
    def __init__(self, model_name: str = "124M", models_dir: str = "models/gpt-2/models"):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.encoder = None
        self.hparams = None
        self.session = None
        
        if GPT2_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load GPT-2 model."""
        try:
            self.encoder = encoder.get_encoder(self.model_name, str(self.models_dir))
            self.hparams = model.default_hparams()
            
            with open(self.models_dir / self.model_name / "hparams.json") as f:
                self.hparams.override_from_dict(json.load(f))
            
            logger.info(f"GPT-2 model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GPT-2 model: {e}")
            GPT2_AVAILABLE = False
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text using GPT-2."""
        if not GPT2_AVAILABLE or not self.encoder:
            return f"[GPT-2 not available] Prompt: {prompt}"
        
        try:
            with tf.Session(graph=tf.Graph()) as sess:
                context = tf.placeholder(tf.int32, [1, None])
                output = sample.sample_sequence(
                    hparams=self.hparams,
                    length=max_length,
                    context=context,
                    batch_size=1,
                    temperature=temperature,
                    top_k=40
                )
                
                saver = tf.train.Saver()
                ckpt = tf.train.latest_checkpoint(self.models_dir / self.model_name)
                saver.restore(sess, ckpt)
                
                context_tokens = self.encoder.encode(prompt)
                out = sess.run(output, feed_dict={context: [context_tokens]})
                generated_text = self.encoder.decode(out[0][len(context_tokens):])
                
                return generated_text
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"[Generation failed] {prompt}"

class EnhancedLiber777Analyzer:
    """Enhanced analyzer with GPT-2 integration."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.gpt2_generator = GPT2TextGenerator()
        
    def analyze_with_gpt2(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze page data with GPT-2 enhancement."""
        
        # Create analysis prompt
        prompt = f"""Analyze this Liber777revised page:

Page Number: {page_data.get('page_number', 'Unknown')}
Content Type: {page_data.get('content_type', 'Unknown')}
Image Dimensions: {page_data.get('image_width', 0)}x{page_data.get('image_height', 0)}
Color Mode: {page_data.get('color_mode', 'Unknown')}
Text Density: {page_data.get('text_density', 0):.4f}
Complexity Score: {page_data.get('complexity_score', 0):.4f}
Visual Elements: {', '.join(page_data.get('visual_elements', []))}
Categories: {', '.join(page_data.get('categories', []))}

Based on this information, provide insights about:
1. What type of content this page likely contains
2. Its significance in the Liber777revised context
3. Potential magical or esoteric elements
4. How it relates to other pages

Analysis:"""
        
        # Generate enhanced analysis
        enhanced_analysis = self.gpt2_generator.generate_text(
            prompt=prompt,
            max_length=200,
            temperature=0.7
        )
        
        # Add to page data
        page_data['gpt2_analysis'] = enhanced_analysis
        page_data['analysis_timestamp'] = datetime.now().isoformat()
        
        return page_data
    
    def generate_symbolic_insights(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Generate symbolic insights using GPT-2."""
        
        prompt = f"""Analyze this Liber777revised dataset:

Total Pages: {dataset.get('total_pages', 0)}
Content Types: {dataset.get('content_types', {})}
Categories: {list(dataset.get('categories', {}).keys())}
Average Quality: {dataset.get('average_quality_score', 0):.4f}

Generate symbolic insights about:
1. The overall structure and organization
2. Potential magical correspondences
3. Hidden patterns and relationships
4. Esoteric significance
5. Recommendations for further analysis

Symbolic Analysis:"""
        
        symbolic_insights = self.gpt2_generator.generate_text(
            prompt=prompt,
            max_length=300,
            temperature=0.8
        )
        
        return {
            'symbolic_insights': symbolic_insights,
            'generation_timestamp': datetime.now().isoformat(),
            'model_used': 'GPT-2',
            'analysis_type': 'symbolic_enhancement'
        }

def main():
    """Main function for GPT-2 enhanced analysis."""
    print("🚀 GPT-2 Enhanced Liber777revised Analysis")
    print("=" * 50)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedLiber777Analyzer(
        root_path=r"C:\\Users\\thoma\\Desktop\\Liber777revised_jp2"
    )
    
    # Load existing analysis
    analysis_file = Path("analysis/liber777_detailed_analysis.json")
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Loaded {len(data.get('pages', []))} pages for GPT-2 enhancement")
        
        # Enhance first few pages with GPT-2
        enhanced_pages = []
        for i, page in enumerate(data.get('pages', [])[:5]):  # Process first 5 pages
            print(f" Enhancing page {i+1}...")
            enhanced_page = analyzer.analyze_with_gpt2(page)
            enhanced_pages.append(enhanced_page)
        
        # Generate symbolic insights
        print("🔮 Generating symbolic insights...")
        symbolic_insights = analyzer.generate_symbolic_insights(data.get('dataset', {}))
        
        # Save enhanced results
        enhanced_data = {
            'original_data': data,
            'enhanced_pages': enhanced_pages,
            'symbolic_insights': symbolic_insights,
            'enhancement_info': {
                'gpt2_model': '124M',
                'enhancement_timestamp': datetime.now().isoformat(),
                'pages_enhanced': len(enhanced_pages)
            }
        }
        
        output_file = Path("analysis/liber777_gpt2_enhanced.json")
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2, default=str)
        
        print(f"💾 Enhanced analysis saved to: {output_file}")
        print("✅ GPT-2 enhancement complete!")
    
    else:
        print("❌ No existing analysis found. Run the basic analysis first.")

if __name__ == "__main__":
    main()
'''
        
        script_path = self.workspace_path / "scripts" / "gpt2_integration.py"
        script_path.parent.mkdir(exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(integration_script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created GPT-2 integration script: {script_path}")
        return script_path

class WorkspaceOrganizer:
    """Organizes the cleaned workspace."""
    
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = Path(workspace_path)
    
    def create_workflow_script(self):
        """Create a streamlined workflow script."""
        
        workflow_script = '''#!/usr/bin/env python3
"""
Streamlined Liber777revised Analysis Workflow
Using Perspective D<cide> Framework + GPT-2

This script provides a complete workflow for analyzing Liber777revised JP2 files.
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

def run_analysis_workflow():
    """Run the complete analysis workflow."""
    
    print("🎯 Liber777revised Analysis Workflow")
    print("=" * 50)
    
    # Step 1: Basic Analysis
    print("📊 Step 1: Running basic analysis...")
    try:
        subprocess.run([sys.executable, "scripts/run_liber777_analysis.py"], check=True)
        print("✅ Basic analysis complete")
    except subprocess.CalledProcessError as e:
        print(f"❌ Basic analysis failed: {e}")
        return
    
    # Step 2: GPT-2 Enhancement
    print("🔮 Step 2: Running GPT-2 enhancement...")
    try:
        subprocess.run([sys.executable, "scripts/gpt2_integration.py"], check=True)
        print("✅ GPT-2 enhancement complete")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  GPT-2 enhancement failed (continuing without it): {e}")
    
    # Step 3: Generate Reports
    print(" Step 3: Generating reports...")
    try:
        # Create summary report
        summary_script = '''
import json
from pathlib import Path

# Load enhanced analysis
analysis_file = Path("analysis/liber777_gpt2_enhanced.json")
if analysis_file.exists():
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    # Create summary
    summary = {
        "total_pages": len(data.get('original_data', {}).get('pages', [])),
        "enhanced_pages": len(data.get('enhanced_pages', [])),
        "symbolic_insights": data.get('symbolic_insights', {}),
        "workflow_completed": True
    }
    
    with open("analysis/workflow_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Workflow summary generated")
else:
    print("❌ Enhanced analysis not found")
'''
        
        with open("temp_summary.py", 'w') as f:
            f.write(summary_script)
        
        subprocess.run([sys.executable, "temp_summary.py"], check=True)
        os.remove("temp_summary.py")
        
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")
    
    print("🎉 Workflow complete!")
    print("📁 Check the 'analysis/' directory for results")

if __name__ == "__main__":
    run_analysis_workflow()
'''
        
        script_path = self.workspace_path / "scripts" / "workflow.py"
        script_path.parent.mkdir(exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(workflow_script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created workflow script: {script_path}")
        return script_path

def main():
    """Main cleanup and integration function."""
    print("🧹 GPT-2 Integration & Workspace Cleanup")
    print("=" * 50)
    
    # Initialize components
    cleaner = WorkspaceCleaner()
    integrator = GPT2Integrator()
    organizer = WorkspaceOrganizer()
    
    # Step 1: Create backup
    print("📦 Step 1: Creating backup...")
    cleaner.create_backup()
    
    # Step 2: Clean GPT-2
    print("🧹 Step 2: Cleaning GPT-2...")
    cleaner.clean_gpt2()
    
    # Step 3: Clean workspace
    print("🧹 Step 3: Cleaning workspace...")
    cleaner.clean_workspace()
    
    # Step 4: Organize workspace
    print(" Step 4: Organizing workspace...")
    cleaner.organize_workspace()
    
    # Step 5: Create integration script
    print("🔗 Step 5: Creating GPT-2 integration...")
    integration_script = integrator.create_integration_script()
    
    # Step 6: Create workflow script
    print("⚡ Step 6: Creating workflow script...")
    workflow_script = organizer.create_workflow_script()
    
    # Step 7: Create new README
    print("📝 Step 7: Creating new README...")
    create_new_readme()
    
    print("\n✅ Cleanup and integration complete!")
    print("\n📋 Next steps:")
    print("1. Run: python scripts/workflow.py")
    print("2. Check the 'analysis/' directory for results")
    print("3. Review the new README.md for usage instructions")
    print("\n🎯 Your workspace is now clean and organized!")

def create_new_readme():
    """Create a new README with updated information."""
    
    readme_content = '''# Liber777revised Analysis System

A comprehensive analysis system for Liber777revised JP2 files using the Perspective D<cide> framework with GPT-2 integration.

## 🚀 Quick Start

```bash
# Run the complete analysis workflow
python scripts/workflow.py
```

## 📁 Directory Structure

```
├── analysis/           # Analysis results
├── models/            # GPT-2 models
├── framework/         # Perspective D<cide> framework
├── docs/             # Documentation
├── scripts/          # Analysis scripts
└── data/             # Data files
```

## 🎯 **How to Use the Cleanup System**

### **1. Run the Cleanup**

```bash
python cleanup_and_integrate.py
```

### **2. What It Does**

- **🧹 Removes garbage**: Deletes unnecessary files, cache, logs, duplicates
- **📁 Organizes workspace**: Creates logical directory structure
- **🔗 Integrates GPT-2**: Connects GPT-2 with your analysis system
- **⚡ Creates workflow**: Streamlined analysis pipeline
- **📝 Updates documentation**: New README and guides

### **3. After Cleanup**

Your workspace will be organized as:

```
├── analysis/           # Analysis results
├── models/            # GPT-2 models (cleaned)
├── framework/         # Perspective D<cide> framework
├── docs/             # Documentation
├── scripts/          # Analysis scripts
├── data/             # Data files
└── backup_before_cleanup/  # Backup of original files
```

### **4. Run the New Workflow**

```bash
python scripts/workflow.py
```

This will:
1. Run basic Liber777revised analysis
2. Enhance with GPT-2 text generation
3. Generate symbolic insights
4. Create comprehensive reports

### **5. Key Benefits**

- **🚀 Faster**: Removed unnecessary files and duplicates
- **🧠 Smarter**: GPT-2 integration for enhanced analysis
- **📊 Better**: Organized output and reporting
- **🧹 Cleaner**: Logical directory structure
- **📈 Scalable**: Easy to extend and modify

The system now combines your dynamic Liber777revised analysis with GPT-2's text generation capabilities to create a powerful, clean, and organized analysis workflow! 🎉
'''
    
    readme_path = Path("README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created new README: {readme_path}")

if __name__ == "__main__":
    main() 