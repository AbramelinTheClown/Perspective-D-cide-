# 🧬 Symbolic Project Mapper (symproj) - Complete Usage Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [CLI Usage](#cli-usage)
3. [Programmatic Usage](#programmatic-usage)
4. [LLM Integration](#llm-integration)
5. [Python Project Analysis](#python-project-analysis)
6. [Advanced Examples](#advanced-examples)
7. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Installation
```bash
# No installation required - it's a standalone library
# Just ensure you have Python 3.9+ and requests (optional for LLM features)
pip install requests
```

### Basic Usage
```bash
# Analyze a React project
python symproj.py scan ./my-react-app --out project_map.json

# Analyze Python project
python simple_python_analyzer.py

# Run test suite
python test_symproj.py
```

## 🖥️ CLI Usage

### Basic Scanning
```bash
# Scan current directory for JS/TS files
python symproj.py scan . --out analysis.json

# Scan specific directory
python symproj.py scan ./my-project --out my_analysis.json

# Include only specific file types
python symproj.py scan . --include "*.js" "*.jsx" "*.ts" "*.tsx"

# Exclude specific directories
python symproj.py scan . --exclude-dirs "node_modules" "dist" "build"
```

### With LLM Enrichment
```bash
# Using OpenAI (requires OPENAI_API_KEY environment variable)
python symproj.py scan . --llm openai --model gpt-4o-mini

# Using LM Studio (local)
python symproj.py scan . --llm lmstudio --api-base http://localhost:1234 --model local/llama-3-8b

# Using Ollama (local)
python symproj.py scan . --llm ollama --model llama3

# Custom API key
python symproj.py scan . --llm openai --model gpt-4o-mini --api-key your-api-key
```

### Advanced Options
```bash
# Read more content for LLM analysis
python symproj.py scan . --read-kb 256

# Custom output file
python symproj.py scan . --out detailed_analysis.json

# Verbose output
python symproj.py scan . --out analysis.json --read-kb 512
```

## 💻 Programmatic Usage

### Basic JavaScript/TypeScript Analysis
```python
from symproj import ProjectScanner, NoopLLM

# Create scanner
scanner = ProjectScanner("./my-react-app", llm=NoopLLM())

# Scan project
project_map = scanner.scan()

# Access results
print(f"Found {len(project_map.nodes)} files")
print(f"File types: {project_map.meta['file_types']}")

# Save to JSON
project_map.save("my_analysis.json")

# Access individual files
for node in project_map.nodes:
    print(f"{node.id}: {node.type}")
    print(f"  Imports: {node.imports}")
    print(f"  Functions: {[f.name for f in node.functions]}")
```

### Python Project Analysis
```python
from simple_python_analyzer import SimplePythonScanner

# Create scanner
scanner = SimplePythonScanner("./my-python-project")

# Scan project
project_map = scanner.scan()

# Access results
print(f"Total files: {len(project_map.nodes)}")
print(f"Total lines: {sum(node.line_count for node in project_map.nodes)}")

# Find largest files
largest_files = sorted(project_map.nodes, key=lambda x: x.line_count, reverse=True)
for file in largest_files[:5]:
    print(f"{file.id}: {file.line_count} lines")
```

### Custom Configuration
```python
from symproj import ProjectScanner, NoopLLM

# Custom scanner configuration
scanner = ProjectScanner(
    root="./my-project",
    llm=NoopLLM(),
    include_patterns=["*.js", "*.jsx", "*.ts", "*.tsx"],
    exclude_dirs=["node_modules", ".git", "dist", "build"],
    read_snippet_kb=256
)

project_map = scanner.scan()
```

## 🤖 LLM Integration

### OpenAI Integration
```python
from symproj import ProjectScanner, OpenAIClient
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create LLM client
llm = OpenAIClient(model="gpt-4o-mini")

# Create scanner with LLM
scanner = ProjectScanner("./my-project", llm=llm)

# Scan with AI enrichment
project_map = scanner.scan()

# Access enriched data
for node in project_map.nodes:
    if node.summary:
        print(f"{node.id}: {node.summary}")
    
    for func in node.functions:
        if func.summary:
            print(f"  {func.name}: {func.summary}")
```

### LM Studio Integration
```python
from symproj import ProjectScanner, LocalChatClient

# Create local LLM client
llm = LocalChatClient(
    base_url="http://localhost:1234",
    model="local/llama-3-8b"
)

# Create scanner
scanner = ProjectScanner("./my-project", llm=llm)
project_map = scanner.scan()
```

### Custom LLM Client
```python
from symproj import LLMClient

class CustomLLMClient(LLMClient):
    def summarize_node(self, node, file_text):
        # Your custom LLM integration
        return "Custom summary", {"function_name": "Custom function summary"}

# Use custom client
scanner = ProjectScanner("./my-project", llm=CustomLLMClient())
project_map = scanner.scan()
```

## 🐍 Python Project Analysis

### Basic Python Analysis
```python
from simple_python_analyzer import SimplePythonScanner

# Analyze Python project
scanner = SimplePythonScanner("./my-python-project")
project_map = scanner.scan()

# Get statistics
total_files = len(project_map.nodes)
total_lines = sum(node.line_count for node in project_map.nodes)
total_functions = sum(len(node.functions) for node in project_map.nodes)
total_classes = sum(len(node.classes) for node in project_map.nodes)

print(f"Files: {total_files}")
print(f"Lines: {total_lines}")
print(f"Functions: {total_functions}")
print(f"Classes: {total_classes}")
```

### File Type Analysis
```python
# Group files by type
files_by_type = {}
for node in project_map.nodes:
    if node.type not in files_by_type:
        files_by_type[node.type] = []
    files_by_type[node.type].append(node)

# Analyze each type
for file_type, files in files_by_type.items():
    print(f"\n{file_type.upper()} files ({len(files)}):")
    for file in files[:3]:  # Show first 3
        print(f"  {file.id}: {file.line_count} lines")
```

### Complexity Analysis
```python
# Find most complex files
complex_files = sorted(project_map.nodes, key=lambda x: len(x.functions), reverse=True)
print("Most complex files:")
for file in complex_files[:5]:
    print(f"  {file.id}: {len(file.functions)} functions")

# Find largest files
large_files = sorted(project_map.nodes, key=lambda x: x.line_count, reverse=True)
print("Largest files:")
for file in large_files[:5]:
    print(f"  {file.id}: {file.line_count} lines")
```

## 🔧 Advanced Examples

### Multi-Project Analysis
```python
import json
from symproj import ProjectScanner, NoopLLM
from simple_python_analyzer import SimplePythonScanner

def analyze_multiple_projects(project_paths):
    results = {}
    
    for path in project_paths:
        print(f"Analyzing {path}...")
        
        # Try JS/TS analysis first
        try:
            js_scanner = ProjectScanner(path, llm=NoopLLM())
            js_map = js_scanner.scan()
            results[path] = {
                "type": "javascript",
                "files": len(js_map.nodes),
                "file_types": js_map.meta["file_types"]
            }
        except:
            # Fall back to Python analysis
            py_scanner = SimplePythonScanner(path)
            py_map = py_scanner.scan()
            results[path] = {
                "type": "python",
                "files": len(py_map.nodes),
                "file_types": py_map.meta["file_types"]
            }
    
    return results

# Usage
projects = ["./project1", "./project2", "./project3"]
results = analyze_multiple_projects(projects)
print(json.dumps(results, indent=2))
```

### Dependency Analysis
```python
def analyze_dependencies(project_map):
    # Build dependency graph
    dependencies = {}
    
    for node in project_map.nodes:
        dependencies[node.id] = {
            "imports": node.imports,
            "links": node.links,
            "type": node.type
        }
    
    # Find files with most dependencies
    most_dependent = sorted(
        dependencies.items(),
        key=lambda x: len(x[1]["links"]),
        reverse=True
    )
    
    print("Files with most dependencies:")
    for file_id, deps in most_dependent[:5]:
        print(f"  {file_id}: {len(deps['links'])} dependencies")
    
    return dependencies
```

### Export to Different Formats
```python
import json
import csv

def export_analysis(project_map, format="json"):
    if format == "json":
        project_map.save("analysis.json")
    
    elif format == "csv":
        with open("analysis.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Type", "Lines", "Functions", "Classes"])
            
            for node in project_map.nodes:
                writer.writerow([
                    node.id,
                    node.type,
                    node.line_count,
                    len(node.functions),
                    len(node.classes)
                ])
    
    elif format == "markdown":
        with open("analysis.md", "w") as f:
            f.write("# Project Analysis\n\n")
            f.write(f"Total files: {len(project_map.nodes)}\n\n")
            
            for node in project_map.nodes:
                f.write(f"## {node.id}\n")
                f.write(f"- Type: {node.type}\n")
                f.write(f"- Lines: {node.line_count}\n")
                f.write(f"- Functions: {len(node.functions)}\n")
                f.write(f"- Classes: {len(node.classes)}\n\n")

# Usage
export_analysis(project_map, "json")
export_analysis(project_map, "csv")
export_analysis(project_map, "markdown")
```

## 🛠️ Troubleshooting

### Common Issues

**1. No files found**
```bash
# Check if files exist
ls -la

# Try with different patterns
python symproj.py scan . --include "*.py"

# Check exclude directories
python symproj.py scan . --exclude-dirs ""
```

**2. LLM integration fails**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test with no LLM first
python symproj.py scan . --llm none

# Check network connectivity
curl https://api.openai.com/v1/models
```

**3. Python analysis includes virtual environment**
```python
# Use specific exclusions
scanner = SimplePythonScanner(
    root="./my-project",
    exclude_dirs=[".venv", "venv", "env", "__pycache__"]
)
```

**4. Memory issues with large projects**
```bash
# Reduce content read for LLM
python symproj.py scan . --read-kb 64

# Exclude large directories
python symproj.py scan . --exclude-dirs "node_modules" "dist" "build"
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed parsing information
scanner = ProjectScanner("./my-project")
project_map = scanner.scan()
```

### Performance Tips
```python
# For large projects, use no LLM first
scanner = ProjectScanner("./large-project", llm=NoopLLM())

# Process in chunks
for chunk in chunks(project_map.nodes, 100):
    # Process chunk
    pass

# Use specific file patterns
scanner = ProjectScanner(
    "./project",
    include_patterns=["*.js", "*.ts"]  # Only specific files
)
```

## 📊 Output Format

### JSON Structure
```json
{
  "nodes": [
    {
      "id": "Component.js",
      "path": "src/components/Component.js",
      "type": "gui",
      "imports": ["📦 react", "🧩 ./utils"],
      "links": ["utils"],
      "functions": [
        {
          "name": "Component",
          "params": ["props"],
          "returns": "JSX.Element",
          "calls": ["useState", "useEffect"],
          "summary": "React component description"
        }
      ],
      "summary": "File description",
      "size_bytes": 1024,
      "last_modified": "1703123456.789"
    }
  ],
  "meta": {
    "root": "/path/to/project",
    "total_files": 15,
    "file_types": {
      "page": 3,
      "hook": 2,
      "button": 2,
      "util": 4,
      "gui": 4
    },
    "scanned_at": "/current/working/directory"
  }
}
```

This comprehensive usage guide covers all the main features and use cases of the symproj library. You can now analyze any project structure and generate detailed insights! 