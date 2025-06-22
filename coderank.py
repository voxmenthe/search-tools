import ast
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx


def path_to_module_fqn(file_path: str, abs_repo_path: str) -> Optional[str]:
    """
    Converts an absolute file path to a fully qualified Python module name.
    e.g., /path/to/repo/pkg/mod.py -> pkg.mod
    e.g., /path/to/repo/pkg/__init__.py -> pkg
    """
    file_path = os.path.normpath(os.path.abspath(file_path))
    abs_repo_path = os.path.normpath(os.path.abspath(abs_repo_path))

    if not file_path.startswith(abs_repo_path):
        return None

    repo_path_for_rel = abs_repo_path
    if abs_repo_path != os.path.dirname(abs_repo_path):
        repo_path_for_rel = abs_repo_path + os.path.sep
    
    try:
        relative_path = os.path.relpath(file_path, abs_repo_path)
    except ValueError:
        return None

    module_path_no_ext, _ = os.path.splitext(relative_path)
    parts = module_path_no_ext.split(os.path.sep)

    if not parts:
        return None

    if parts[-1] == "__init__":
        parts.pop()
        if not parts:
            return None 
    
    return ".".join(parts) if parts else None


def resolve_relative_import(current_module_fqn: str, level: int, module_in_from_statement: Optional[str]) -> Optional[str]:
    """
    Resolves a relative import to a fully qualified name.
    """
    if not current_module_fqn:
        if level > 0 and module_in_from_statement:
            return module_in_from_statement
        return None

    path_parts = current_module_fqn.split('.')
    
    if level > len(path_parts):
        return None
    
    base_module_parts = path_parts[:-level]

    if module_in_from_statement:
        resolved_parts = base_module_parts + module_in_from_statement.split('.')
        return ".".join(resolved_parts)
    else:
        if not base_module_parts:
            return None
        return ".".join(base_module_parts)


def get_imports_from_file(file_path: str, current_module_fqn: str) -> Set[str]:
    """
    Parses a Python file and extracts its imports as fully qualified names.
    """
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        tree = ast.parse(content, filename=file_path)
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        
        elif isinstance(node, ast.ImportFrom):
            module_name_in_from = node.module
            level = node.level

            if level > 0:  # Relative import
                resolved_base = resolve_relative_import(current_module_fqn, level, module_name_in_from)
                if not resolved_base:
                    continue

                if module_name_in_from:
                    imports.add(resolved_base)
                else:
                    for alias in node.names:
                        imports.add(f"{resolved_base}.{alias.name}")
            
            else:  # Absolute import
                if module_name_in_from:
                    imports.add(module_name_in_from)
    return imports


def extract_python_symbols(file_path: str, current_module_fqn: str) -> Dict[str, Dict[str, str]]:
    """
    Extracts modules, classes, functions, and methods FQNs from a Python file.
    Returns a dict of symbol_fqn -> {type, module_fqn, file_path}
    """
    symbols = {}
    
    if not current_module_fqn:
        return symbols

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        tree = ast.parse(content, filename=file_path)
    except Exception:
        return symbols

    # Add module symbol itself
    symbols[current_module_fqn] = {
        "type": "module",
        "module_fqn": current_module_fqn,
        "file_path": file_path,
    }

    # Process top-level items
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_fqn = f"{current_module_fqn}.{node.name}"
            symbols[class_fqn] = {
                "type": "class",
                "module_fqn": current_module_fqn,
                "file_path": file_path,
            }
            # Extract methods within this class
            for sub_node in node.body:
                if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_fqn = f"{class_fqn}.{sub_node.name}"
                    symbols[method_fqn] = {
                        "type": "method",
                        "module_fqn": current_module_fqn,
                        "file_path": file_path,
                    }
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_fqn = f"{current_module_fqn}.{node.name}"
            symbols[function_fqn] = {
                "type": "function",
                "module_fqn": current_module_fqn,
                "file_path": file_path,
            }
    
    return symbols


def discover_python_files(repo_path: str) -> List[str]:
    """Finds all .py files in the repository."""
    py_files = []
    abs_repo_path = os.path.abspath(repo_path)
    for root, _, files in os.walk(abs_repo_path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def discover_markdown_files(repo_path: str) -> List[str]:
    """Finds all .md and .markdown files in the repository."""
    md_files = []
    abs_repo_path = os.path.abspath(repo_path)
    for root, _, files in os.walk(abs_repo_path):
        for file in files:
            if file.endswith(".md") or file.endswith(".markdown"):
                md_files.append(os.path.join(root, file))
    return md_files


def analyze_markdown_file_references(md_file_path: str, all_python_fqns: Set[str], python_symbols_db: Dict) -> Set[str]:
    """
    Analyzes a Markdown file to find references to Python symbols.
    Returns a set of module FQNs that are referenced in the Markdown file.
    """
    referenced_module_fqns_in_md = set()
    try:
        with open(md_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return referenced_module_fqns_in_md

    for py_fqn in all_python_fqns:
        escaped_py_fqn = re.escape(py_fqn)
        try:
            if re.search(r'\b' + escaped_py_fqn + r'\b', content):
                symbol_info = python_symbols_db.get(py_fqn)
                if symbol_info and "module_fqn" in symbol_info:
                    referenced_module_fqns_in_md.add(symbol_info["module_fqn"])
                elif py_fqn in python_symbols_db and python_symbols_db[py_fqn]["type"] == "module":
                    referenced_module_fqns_in_md.add(py_fqn)
        except re.error:
            pass

    return referenced_module_fqns_in_md


def calculate_coderank(
    repo_path: str,
    external_modules: List[str] = None,
    damping_factor: float = 0.85,
    weight_internal: float = 1.0,
    weight_external_import: float = 0.5,
    weight_external_dependency: float = 0.5,
    analyze_markdown: bool = False
) -> Dict[str, any]:
    """
    Main analysis function that calculates CodeRank scores.
    
    Returns a dictionary with:
    - module_ranks: Dict[module_fqn, score]
    - markdown_ranks: Dict[md_file_path, score] (if analyze_markdown=True)
    - module_map: Dict[module_fqn, file_path]
    - python_symbols_db: Dict[symbol_fqn, symbol_info]
    - import_graph: networkx.DiGraph
    """
    if external_modules is None:
        external_modules = ["numpy", "pandas", "sklearn", "torch", "tensorflow", "requests", "django", "flask"]
    
    abs_repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(abs_repo_path):
        raise ValueError(f"Repository path {abs_repo_path} not found or not a directory.")

    specified_external_modules = set(external_modules)
    
    # Discover Python files
    all_py_files = discover_python_files(abs_repo_path)
    if not all_py_files:
        return {
            "module_ranks": {},
            "markdown_ranks": {},
            "module_map": {},
            "python_symbols_db": {},
            "import_graph": nx.DiGraph()
        }

    module_map = {}  # FQN -> file_path
    file_to_fqn = {}  # file_path -> FQN
    internal_module_fqns = set()
    python_symbols_db = {}  # Stores FQNs for modules, classes, functions, methods

    # Map files to module names and extract symbols
    for f_path in all_py_files:
        fqn = path_to_module_fqn(f_path, abs_repo_path)
        if fqn:
            module_map[fqn] = f_path
            file_to_fqn[f_path] = fqn
            internal_module_fqns.add(fqn)
            # Extract symbols for this file
            symbols = extract_python_symbols(f_path, fqn)
            python_symbols_db.update(symbols)
    
    # Initialize graphs
    G_imports = nx.DiGraph()  # A imports B: A -> B
    G_imported_by = nx.DiGraph()  # A imports B: B -> A (for PageRank "outgoing")

    all_graph_nodes = set(internal_module_fqns)
    for ext_mod in specified_external_modules:
        all_graph_nodes.add(ext_mod)

    # Add all potential nodes
    for node_fqn in all_graph_nodes:
        G_imports.add_node(node_fqn)
        G_imported_by.add_node(node_fqn)

    # Parse imports and build dependency graphs
    for f_path in all_py_files:
        current_fqn = file_to_fqn.get(f_path)
        if not current_fqn:
            continue

        imported_fqns = get_imports_from_file(f_path, current_fqn)
        
        for imported_fqn_full in imported_fqns:
            target_node_fqn = None
            is_external = False
            weight_to = weight_internal
            weight_from = weight_internal

            # Check if it's a specified external module
            imported_root = imported_fqn_full.split('.')[0]
            if imported_root in specified_external_modules:
                target_node_fqn = imported_root
                is_external = True
                weight_to = weight_external_import
                weight_from = weight_external_dependency
            # Check if it's an internal module
            elif imported_fqn_full in internal_module_fqns:
                target_node_fqn = imported_fqn_full

            if target_node_fqn and current_fqn != target_node_fqn:
                G_imports.add_edge(current_fqn, target_node_fqn, weight=weight_to)
                G_imported_by.add_edge(target_node_fqn, current_fqn, weight=weight_from)
    
    # Calculate PageRank
    try:
        pagerank_being_imported = nx.pagerank(G_imports, alpha=damping_factor, weight='weight', tol=1.0e-8, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        pagerank_being_imported = {node: 1.0 / len(G_imports) for node in G_imports.nodes()}

    try:
        pagerank_importing_others = nx.pagerank(G_imported_by, alpha=damping_factor, weight='weight', tol=1.0e-8, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        pagerank_importing_others = {node: 1.0 / len(G_imported_by) for node in G_imported_by.nodes()}

    # Calculate final CodeRank scores
    code_ranks = {}
    for module_fqn in internal_module_fqns:
        score_being_imported = pagerank_being_imported.get(module_fqn, 0)
        score_importing_others = pagerank_importing_others.get(module_fqn, 0)
        code_ranks[module_fqn] = score_being_imported + score_importing_others

    # Markdown analysis
    markdown_ranks = {}
    if analyze_markdown:
        md_files = discover_markdown_files(abs_repo_path)
        if md_files:
            all_py_fqns_for_md_search = set(python_symbols_db.keys())
            markdown_to_referenced_modules = defaultdict(set)
            
            for md_file in md_files:
                referenced_modules = analyze_markdown_file_references(md_file, all_py_fqns_for_md_search, python_symbols_db)
                if referenced_modules:
                    markdown_to_referenced_modules[md_file].update(referenced_modules)
            
            # Rank Markdown files
            for md_file, referenced_mods in markdown_to_referenced_modules.items():
                score = sum(code_ranks.get(mod_fqn, 0) for mod_fqn in referenced_mods)
                markdown_ranks[md_file] = score

    return {
        "module_ranks": code_ranks,
        "markdown_ranks": markdown_ranks,
        "module_map": module_map,
        "python_symbols_db": python_symbols_db,
        "import_graph": G_imports,
        "internal_modules": internal_module_fqns,
        "external_modules": specified_external_modules
    }