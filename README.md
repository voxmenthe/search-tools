# 🔍 Search Tools MCP Server

> ⚡ An intelligent Model Context Protocol (MCP) server that supercharges code analysis with advanced search capabilities and dependency mapping

## 🌟 Overview

The **Search Tools MCP Server** is a powerful toolkit that combines traditional code search with intelligent analysis algorithms. It leverages the **CodeRank** algorithm (inspired by PageRank) to identify the most critical modules in your codebase and provides sophisticated search capabilities that go beyond simple text matching.

## 🎯 Key Features

### 🔎 **Smart Search Capabilities**
- **Contextual Keyword Search**: Ripgrep-powered search with configurable context lines
- **Symbol Discovery**: Extract and analyze functions, classes, methods, and modules
- **Usage Tracking**: Find where symbols are used across your codebase
- **Priority-Ranked Results**: Search results ranked by code importance

### 🧠 **Intelligence & Analysis**
- **CodeRank Algorithm**: Identify the most critical modules using network analysis
- **Dependency Mapping**: Trace complex dependency chains and impact analysis
- **Hotspot Detection**: Find code areas that are both highly connected and frequently used
- **Refactoring Impact**: Analyze the potential impact of code changes

### 🎨 **Advanced Filtering**
- Symbol type filtering (functions, methods, classes)
- File inclusion/exclusion patterns
- External module dependency tracking
- Markdown documentation analysis

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- `uv` package manager
- `kit` CLI tool (for symbol analysis)
- `ripgrep` (for fast text search)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd search-tools

# Install dependencies
uv sync
```

## ⚙️ Configuration

### Adding to Cursor/Windsurf

Add the following configuration to your `mcp.json` file:

```json
{
  "mcpServers": {
    "search-tools": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--directory",
        "/path/to/search-tools",
        "main.py"
      ]
    }
  }
}
```

**For macOS users with Homebrew:**
```json
{
  "mcpServers": {
    "search-tools": {
      "command": "/Users/yourusername/.local/bin/uv",
      "args": [
        "run",
        "--directory",
        "/path/to/your/search-tools/directory",
        "main.py"
      ]
    }
  }
}
```

### 📍 Finding Your Paths

To find the correct paths for your system:

```bash
# Find uv location
which uv

# Get absolute path to search-tools directory  
pwd  # (run this from the search-tools directory)
```

## 🚀 Available Tools

### 🔍 `contextual_keyword_search`
Search for keywords with configurable context lines around matches.

**Parameters:**
- `keyword`: Search term (case insensitive)
- `working_directory`: Absolute path to search directory
- `num_context_lines`: Lines of context (default: 2)

### 🏗️ `get_repo_symbols`
Extract symbols (functions, classes, methods) from your codebase.

**Parameters:**
- `repo`: Repository path
- `working_directory`: Command execution directory
- `keep_types`: Filter by symbol types
- `file_must_contain/file_must_not_contain`: File filtering

### 📊 `get_symbol_usages`
Find where specific symbols are used throughout your codebase.

**Parameters:**
- `repo`: Repository path
- `symbol_name_or_substring`: Symbol to search for
- `working_directory`: Command execution directory
- `symbol_type`: Optional type filter

### 🎯 `coderank_analysis`
Analyze repository importance using the CodeRank algorithm.

**Parameters:**
- `repo_path`: Repository to analyze
- `external_modules`: Comma-separated external dependencies
- `top_n`: Number of top modules to return (default: 10)
- `analyze_markdown`: Include markdown files
- `output_format`: "summary", "detailed", or "json"

### 🔥 `find_code_hotspots`
Identify critical code areas combining connectivity and usage frequency.

**Parameters:**
- `repo_path`: Repository path
- `working_directory`: Command execution directory
- `min_connections`: Minimum import connections (default: 5)
- `include_external`: Include external dependencies
- `top_n`: Number of hotspots to return (default: 20)

### 🌐 `trace_dependency_impact`
Trace dependency chains and analyze refactoring impact.

**Parameters:**
- `repo_path`: Repository path
- `target_module`: Module to analyze
- `working_directory`: Command execution directory
- `analysis_type`: "dependency", "refactoring", or "both"
- `max_depth`: Maximum trace depth (default: 3)
- `change_type`: "modify", "split", "merge", or "remove"

### 🎪 `smart_code_search`
Enhanced search combining ripgrep with CodeRank prioritization.

**Parameters:**
- `keyword`: Search term (supports regex)
- `repo_path`: Repository path
- `working_directory`: Command execution directory
- `rank_results`: Sort by module importance
- `context_lines`: Context lines around matches (default: 3)
- `max_results`: Maximum results to return (default: 20)

## 🧪 Development & Testing

### Running the Server
```bash
# Development mode
uv run mcp dev main.py

# Testing with MCP Inspector
npx @modelcontextprotocol/inspector python main.py
```

### 🔧 Dependencies
- **mcp[cli]**: Model Context Protocol framework
- **cased-kit**: Symbol analysis toolkit
- **networkx**: Graph analysis for CodeRank algorithm

## 🎨 Algorithm Details

### CodeRank Algorithm
The CodeRank algorithm treats your codebase as a directed graph where:
- **Nodes**: Python modules, classes, functions, methods
- **Edges**: Import relationships and dependencies
- **Weights**: Different weights for internal vs external dependencies

This creates a ranking system that identifies the most "central" and important parts of your codebase, similar to how PageRank identifies important web pages.

## 💡 Use Cases

- **🔍 Code Exploration**: Quickly understand large codebases
- **🏗️ Refactoring Planning**: Identify high-impact areas before changes
- **📚 Documentation**: Find the most important modules to document first
- **🐛 Bug Investigation**: Focus on critical code paths
- **👥 Code Review**: Prioritize review efforts on important modules

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source. Please check the license file for details.

---

<div align="center">

**🔮 Powered by the CodeRank Algorithm & Model Context Protocol**

*Making code search intelligent, one repository at a time*

</div>
