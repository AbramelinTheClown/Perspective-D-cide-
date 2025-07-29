# 🧬 Symbolic Project Mapper (symproj)

A comprehensive, reusable Python library for analyzing project structure and building symbolic JSON representations. Perfect for understanding codebases, generating documentation, and enabling AI-powered code analysis.

## 🌟 Features

- **🔍 Intelligent File Type Detection**: Automatically categorizes files as pages, hooks, buttons, utilities, or GUI components
- **📦 Import Analysis**: Extracts and categorizes imports (external packages vs local files)
- **🔗 Dependency Mapping**: Links local imports to actual file nodes
- **⚡ Function Extraction**: Parses function signatures, parameters, and call relationships
- **🤖 LLM Integration**: Optional AI-powered summarization and call-graph analysis
- **📊 Rich Metadata**: File sizes, modification times, and comprehensive statistics
- **🎯 Multiple LLM Support**: OpenAI, LM Studio, Ollama, and other local chat APIs
- **🛠️ CLI Interface**: Easy command-line usage for quick project analysis

## 🚀 Quick Start

### Basic Usage

```bash
# Scan a React project without LLM enrichment
python symproj.py scan ./my-react-app --out project_map.json

# Scan with OpenAI enrichment
python symproj.py scan ./my-react-app --out project_map.json --llm openai --model gpt-4o-mini

# Scan with local LM Studio
python symproj.py scan ./my-react-app --out project_map.json --llm lmstudio --api-base http://localhost:1234 --model local/llama-3-8b
```

### Programmatic Usage

```python
from symproj import ProjectScanner, NoopLLM, OpenAIClient

# Basic scanning
scanner = ProjectScanner("./my-react-app", llm=NoopLLM())
project_map = scanner.scan()

# With LLM enrichment
llm = OpenAIClient(model="gpt-4o-mini")
scanner = ProjectScanner("./my-react-app", llm=llm)
project_map = scanner.scan()

# Save results
project_map.save("project_map.json")
```

## 📋 Output Format

The library generates a comprehensive JSON structure:

```json
{
  "nodes": [
    {
      "id": "Home.js",
      "path": "src/pages/Home.js",
      "type": "page",
      "imports": ["📦 react", "🧩 ./components/PrimaryButton", "🧩 ./hooks/useUserData"],
      "links": ["PrimaryButton", "useUserData"],
      "functions": [
        {
          "name": "Home",
          "params": [],
          "returns": "JSX.Element",
          "calls": ["useEffect", "useUserData", "fetchFeaturedItems"],
          "summary": "Main homepage component that renders user data and featured items"
        }
      ],
      "summary": "React page component for the homepage with user data integration",
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

## 🔧 Configuration

### File Type Detection

The library uses intelligent heuristics to categorize files:

- **`page`**: Files in `/pages/` directories or ending with `.page.tsx`
- **`hook`**: Files starting with "use" or in `/hooks/` directories
- **`button`**: Files containing "button" in the name
- **`gui`**: Files in `/components/` or `/ui/` directories
- **`util`**: Files in `/utils/`, `/lib/`, `/helpers/`, `/services/`, or `/api/` directories

### Import Categorization

- **`📦 external`**: External package imports (e.g., "react", "lodash")
- **`🧩 local`**: Local file imports (e.g., "./components/Button")

## 🤖 LLM Integration

### Supported Providers

1. **OpenAI**: `--llm openai --model gpt-4o-mini`
2. **LM Studio**: `--llm lmstudio --api-base http://localhost:1234 --model local/llama-3-8b`
3. **Ollama**: `--llm ollama --model llama3`
4. **Custom**: Any OpenAI-compatible API endpoint

### LLM Capabilities

When LLM integration is enabled, the library can:

- Generate file summaries explaining the purpose and functionality
- Analyze function behavior and relationships
- Identify code patterns and architectural decisions
- Provide insights about data flow and dependencies

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Custom API base (for local models)
export OPENAI_API_BASE="http://localhost:1234"
```

## 🛠️ Advanced Usage

### Custom File Patterns

```bash
# Include only TypeScript files
python symproj.py scan ./my-project --include "*.ts" "*.tsx"

# Exclude test files
python symproj.py scan ./my-project --exclude-dirs "tests" "__tests__" "node_modules"
```

### Programmatic Customization

```python
from symproj import ProjectScanner, NoopLLM

# Custom configuration
scanner = ProjectScanner(
    root="./my-project",
    llm=NoopLLM(),
    include_patterns=["*.js", "*.jsx", "*.ts", "*.tsx"],
    exclude_dirs=["node_modules", ".git", "dist", "build"],
    read_snippet_kb=256  # Read more content for LLM analysis
)

project_map = scanner.scan()
```

### Custom LLM Client

```python
from symproj import LLMClient

class CustomLLMClient(LLMClient):
    def summarize_node(self, node, file_text):
        # Your custom LLM integration
        return "Custom summary", {"function_name": "Custom function summary"}
```

## 📊 Use Cases

### 1. Codebase Understanding
- Generate visual dependency graphs
- Identify architectural patterns
- Map data flow between components

### 2. Documentation Generation
- Auto-generate component documentation
- Create dependency matrices
- Build interactive code explorers

### 3. AI-Powered Analysis
- Feed project structure to LLMs for code review
- Generate architectural recommendations
- Identify refactoring opportunities

### 4. Project Migration
- Analyze legacy codebases
- Plan migration strategies
- Track architectural evolution

### 5. Team Onboarding
- Create project overviews for new developers
- Generate learning paths through codebase
- Document architectural decisions

## 🔗 Integration Examples

### With Graph Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create dependency graph
G = nx.DiGraph()
for node in project_map.nodes:
    G.add_node(node.id, type=node.type)
    for link in node.links:
        G.add_edge(node.id, link)

# Visualize
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
plt.show()
```

### With Documentation Generators

```python
# Generate Markdown documentation
for node in project_map.nodes:
    print(f"## {node.id}")
    print(f"**Type:** {node.type}")
    print(f"**Summary:** {node.summary or 'No summary available'}")
    print(f"**Dependencies:** {', '.join(node.links)}")
    print()
```

### With MCP (Model Context Protocol)

```python
# Expose project structure as MCP tools
def get_project_structure():
    return project_map.to_json()

def search_components(query):
    return [node for node in project_map.nodes if query.lower() in node.id.lower()]
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_symproj.py
```

This will:
- Create a sample React project
- Test basic scanning functionality
- Test JSON output format
- Test LLM enrichment (if available)
- Generate a sample output file

## 📦 Installation

### Requirements

- Python 3.9+
- Optional: `requests` for LLM integration

### Setup

```bash
# Clone or download symproj.py
# No installation required - it's a standalone library

# Optional: Install requests for LLM features
pip install requests
```

## 🔮 Future Enhancements

- **Tree-sitter Integration**: More accurate AST parsing
- **Language Support**: Python, Java, Go, Rust, and more
- **Visual Interface**: Web-based project explorer
- **Real-time Monitoring**: Watch for file changes
- **Advanced Analytics**: Complexity metrics, code smells detection
- **Integration APIs**: GitHub, GitLab, Bitbucket integration

## 🤝 Contributing

This library is designed to be extensible. Key areas for contribution:

1. **Language Support**: Add parsers for new programming languages
2. **LLM Providers**: Integrate additional AI services
3. **Visualization**: Create better graph and chart outputs
4. **Performance**: Optimize for large codebases
5. **Documentation**: Improve examples and use cases

## 📄 License

This library is provided as-is for educational and development purposes. Feel free to modify and extend for your specific needs.

## 🎯 Example Output

Here's what a typical analysis produces:

```json
{
  "nodes": [
    {
      "id": "Home.js",
      "type": "page",
      "imports": ["📦 react", "🧩 ./components/PrimaryButton"],
      "links": ["PrimaryButton"],
      "functions": [
        {
          "name": "Home",
          "summary": "Main homepage component that displays user data and featured content"
        }
      ],
      "summary": "React page component for the homepage with user authentication and content display"
    }
  ],
  "meta": {
    "total_files": 15,
    "file_types": {"page": 3, "hook": 2, "button": 2, "util": 4, "gui": 4}
  }
}
```

This symbolic representation enables powerful code analysis, documentation generation, and AI-powered insights into your codebase structure and relationships. 