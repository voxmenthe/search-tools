import os
import re
import subprocess
import json
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Tuple
from type_definitions import ModuleChangeData, CochangeData, ContributorData, PropagationData, create_module_change_data, create_cochange_data, create_contributor_data, create_propagation_data
from collections import defaultdict
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP
from coderank import calculate_coderank


mcp = FastMCP()

@mcp.tool()
def contextual_keyword_search(keyword: str, working_directory: str, num_context_lines: int = 2) -> str:
    """
    Search for a keyword in the current directory (wrapper around ripgrep).

    Args:
        keyword (str): The keyword to search for (case insensitive).
        working_directory (str): The directory to search in. Use full absolute path.
        num_context_lines (int): The number of lines of context to return (both before and after the keyword). Default is 2.


    Returns:
        str: The file path and the lines of context around the keyword.
    """
    # use ripgrep to search for the keyword in the current directory
    result = subprocess.run(
        [
            "rg", "-n", "-i", 
            "-B", str(num_context_lines), "-A", str(num_context_lines), 
            keyword, working_directory
        ], 
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout
    else:
        return f"No results found for keyword: {keyword}"


@mcp.tool()
def get_repo_symbols(
    repo: str,
    working_directory: str,
    keep_types: Optional[Iterable[str]] = None,
    file_must_contain: Optional[str] = None,
    file_must_not_contain: Optional[str] = None
) -> List[str] | str:
    """
    Run `kit symbols <repo>` and keep the header plus rows
    that satisfy the filters.

    Parameters
    ----------
    repo : str | Path
        Path or name passed to `kit symbols`.
    working_directory : str
        The directory to run the command from. Use full absolute path.
    keep_types : Iterable[str] | None
        Exact values allowed in the **Type** column
        (e.g. {"function", "method", "class"}).  None ⇒ no type filter.
    file_must_contain : str | None
        post-filter: Keep only rows whose **File** column *contains* this substring.
        None ⇒ no inclusion filter.
    file_must_not_contain : str | None
        post-filter: Discard rows whose **File** column contains this substring.
        None ⇒ no exclusion filter.


    Returns
    -------
    list[str] | str
        Filtered output, ready to `print()` or write to a file.
    """
    # 1) Run the external command
    
    result = subprocess.run(
        ["kit", "symbols", str(repo)],
        check=True,
        text=True,
        capture_output=True,
        cwd=working_directory  # This ensures the command runs in the specified directory
    )
    raw_lines = result.stdout.splitlines(keepends=False)

    # 2) Prepare filters
    keep_types   = set(keep_types) if keep_types else None
    inc_substr   = file_must_contain or ""
    exc_substr   = file_must_not_contain or ""

    # Regex:     split on 2-or-more spaces/tabs
    splitter = re.compile(r"\s{2,}")

    filtered: List[str] = []
    header_passed = False

    for line in raw_lines:
        # Always keep the header (first line) and the separator (second line)
        if not header_passed:
            filtered.append(line)
            # the separator is a run of dashes:  ------
            if re.match(r"-{3,}", line):
                header_passed = True
            continue

        if not line.strip():           # skip blank lines
            continue

        parts = splitter.split(line)
        if len(parts) < 4:
            # Unexpected layout—keep it unchanged.
            filtered.append(line)
            continue

        _, symbol_type, file_col, _ = parts[:4]

        # --- Apply filters ---------------------------------------------------
        if keep_types and symbol_type not in keep_types:
            continue
        if inc_substr   and inc_substr   not in file_col:
            continue
        if exc_substr   and exc_substr   in  file_col:
            continue

        filtered.append(line)

    return filtered


@mcp.tool()
def get_symbol_usages(
    repo: str,
    symbol_name_or_substring: str,
    working_directory: str,
    symbol_type: Optional[str] = None,
    file_must_contain: Optional[str] = None,
    file_must_not_contain: Optional[str] = None,
) -> List[str] | str:
    """
    Run `kit usages <repo> <symbol_name>` and optionally filter by symbol type
    at the CLI level and then post-filter the rows by file inclusion/exclusion
    substrings.

    Parameters
    ----------
    repo : str | Path
        Path or name passed to `kit usages`. Use full absolute path.
    symbol_name_or_substring : str
        The symbol whose usages we want to inspect, or a substring of the symbol name (which can be used to find multiple symbols which share a naming convention).
    working_directory : str
        Directory from which to run the command (absolute path).
    symbol_type : str | None
        "function" or "method" or "class"
    file_must_contain : str | None
        post-filter: Keep only rows whose **File** column *contains* this substring.
        None ⇒ no inclusion filter.
    file_must_not_contain : str | None
        post-filter: Discard rows whose **File** column contains this substring.
        None ⇒ no exclusion filter.

    Returns
    -------
    list[str] | str
        Filtered output, ready to `print()` or write to a file.
    """

    # 1) Build the CLI invocation
    cmd: List[str] = ["kit", "usages", str(repo), symbol_name_or_substring]
    if symbol_type:
        cmd.extend(["--symbol-type", symbol_type])

    # 2) Run the external command
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        cwd=working_directory,
    )
    raw_lines = result.stdout.splitlines(keepends=False)

    # 3) Prepare filters
    inc_substr = file_must_contain or ""
    exc_substr = file_must_not_contain or ""

    splitter = re.compile(r"\s{2,}")  # split on 2-or-more whitespace blocks

    filtered: List[str] = []
    header_passed = False

    for line in raw_lines:
        # Always keep the header (first line) and the separator (second line)
        if not header_passed:
            filtered.append(line)
            if re.match(r"-{3,}", line):
                header_passed = True
            continue

        if not line.strip():  # skip blank lines
            continue

        parts = splitter.split(line)
        # Defensive: expect at least a File column in position 2 or 3 depending on layout.
        file_col: str
        if len(parts) >= 3:
            file_col = parts[2]
        elif len(parts) >= 2:
            file_col = parts[1]
        else:
            filtered.append(line)
            continue

        # --- Apply filters ---------------------------------------------------
        if inc_substr and inc_substr not in file_col:
            continue
        if exc_substr and exc_substr in file_col:
            continue

        filtered.append(line)

    return filtered

# ===============================================
# CodeRank Analysis
# ===============================================

@mcp.tool()
def coderank_analysis(
    repo_path: str,
    external_modules: str = "google,genai,langchain,langgraph,dspy,agn,torch,numpy",
    top_n: int = 10,
    analyze_markdown: bool = False,
    output_format: str = "summary"
) -> str:
    """
    Analyze repository importance using CodeRank algorithm to identify critical modules.
    
    Args:
        repo_path: Path to the repository (absolute path)
        external_modules: Comma-separated list of external modules to track
        top_n: Number of top modules to return
        analyze_markdown: Include markdown files in analysis
        output_format: "summary" for key results, "detailed" for full analysis, "json" for machine-readable
    
    Returns:
        Ranked list of most important modules with scores
    """
    try:
        # Parse external modules
        external_module_list = [m.strip() for m in external_modules.split(',') if m.strip()]
        
        # Run coderank analysis
        results = calculate_coderank(
            repo_path=repo_path,
            external_modules=external_module_list,
            analyze_markdown=analyze_markdown
        )
        
        # Sort module ranks
        sorted_module_ranks = sorted(
            results["module_ranks"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Sort markdown ranks if available
        sorted_markdown_ranks = []
        if analyze_markdown and results["markdown_ranks"]:
            sorted_markdown_ranks = sorted(
                results["markdown_ranks"].items(),
                key=lambda x: x[1],
                reverse=True
            )
        
        if output_format == "json":
            # Return JSON format
            output = {
                "repo_path": repo_path,
                "external_modules": external_module_list,
                "top_modules": [
                    {"module": module, "score": score}
                    for module, score in sorted_module_ranks[:top_n]
                ],
                "total_modules": len(results["module_ranks"])
            }
            
            if analyze_markdown:
                output["top_markdown_files"] = [
                    {"file": os.path.basename(file), "path": file, "score": score}
                    for file, score in sorted_markdown_ranks[:top_n]
                ]
            
            return json.dumps(output, indent=2)
        
        elif output_format == "summary":
            # Build summary output
            lines = ["=== CodeRank Analysis Summary ===\n"]
            lines.append(f"Repository: {repo_path}")
            lines.append(f"Total modules analyzed: {len(results['module_ranks'])}")
            lines.append(f"External modules tracked: {', '.join(external_module_list)}\n")
            lines.append("Top modules by importance:")
            lines.append(f"{'Module'.ljust(40)} | {'CodeRank Score'.rjust(14)}")
            lines.append("-" * 57)
            
            for module, score in sorted_module_ranks[:top_n]:
                lines.append(f"{module[:40].ljust(40)} | {score:14.6f}")
            
            if analyze_markdown and sorted_markdown_ranks:
                lines.append("\n=== Top Markdown Files ===")
                lines.append(f"{'File'.ljust(30)} | {'Score'.rjust(14)}")
                lines.append("-" * 47)
                for file, score in sorted_markdown_ranks[:top_n]:
                    filename = os.path.basename(file)[:30]
                    lines.append(f"{filename.ljust(30)} | {score:14.6f}")
            
            return '\n'.join(lines)
        
        else:  # detailed
            # Build detailed output
            lines = ["=== CodeRank Detailed Analysis ===\n"]
            lines.append(f"Repository: {repo_path}")
            lines.append(f"Total modules: {len(results['module_ranks'])}")
            lines.append(f"Total symbols extracted: {len(results['python_symbols_db'])}")
            lines.append(f"External modules: {', '.join(external_module_list)}\n")
            
            lines.append("All modules ranked by importance:")
            lines.append(f"{'Rank'.rjust(5)} | {'Module'.ljust(50)} | {'Score'.rjust(14)} | {'File Path'}")
            lines.append("-" * 120)
            
            for i, (module, score) in enumerate(sorted_module_ranks, 1):
                file_path = results["module_map"].get(module, "N/A")
                rel_path = os.path.relpath(file_path, repo_path) if file_path != "N/A" else "N/A"
                lines.append(f"{i:5d} | {module[:50].ljust(50)} | {score:14.6f} | {rel_path}")
            
            if analyze_markdown and sorted_markdown_ranks:
                lines.append("\n=== All Markdown Files Ranked ===")
                lines.append(f"{'Rank'.rjust(5)} | {'File'.ljust(40)} | {'Score'.rjust(14)} | {'Full Path'}")
                lines.append("-" * 100)
                for i, (file, score) in enumerate(sorted_markdown_ranks, 1):
                    filename = os.path.basename(file)[:40]
                    lines.append(f"{i:5d} | {filename.ljust(40)} | {score:14.6f} | {file}")
            
            return '\n'.join(lines)
            
    except Exception as e:
        return f"Error in coderank_analysis: {str(e)}"


@mcp.tool()
def find_code_hotspots(
    repo_path: str,
    working_directory: str,
    min_connections: int = 5,
    include_external: bool = True,
    top_n: int = 20
) -> str:
    """
    Identify code hotspots by combining CodeRank with symbol usage frequency.
    Uses kit usages + coderank to find modules that are both highly connected
    and frequently used.
    
    Args:
        repo_path: Repository to analyze
        working_directory: Working directory for commands (absolute path)
        min_connections: Minimum import connections to consider
        include_external: Include external module dependencies
        top_n: Number of top hotspots to return
    
    Returns:
        Hotspot analysis with modules ranked by importance and usage
    """
    try:
        # Get CodeRank analysis
        external_modules = ["numpy", "pandas", "sklearn", "torch", "tensorflow", "requests", "django", "flask"] if include_external else []
        
        coderank_results = calculate_coderank(
            repo_path=repo_path,
            external_modules=external_modules
        )
        
        module_scores = coderank_results["module_ranks"]
        module_map = coderank_results["module_map"]
        
        # Get all symbols from the repository
        symbols_output = get_repo_symbols(
            repo=repo_path,
            working_directory=working_directory,
            keep_types=["function", "class", "method"]
        )
        
        # Parse symbols to count per module
        module_symbol_counts = defaultdict(int)
        module_symbols = defaultdict(list)
        
        for line in symbols_output:
            if '|' in line and not line.startswith('-') and 'Symbol' not in line:
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 4:
                    symbol_name = parts[0]
                    symbol_type = parts[1]
                    file_path = parts[2]
                    
                    # Convert file path to module name
                    abs_file_path = os.path.join(working_directory, file_path) if not os.path.isabs(file_path) else file_path
                    module_fqn = path_to_module_fqn(abs_file_path, repo_path)
                    
                    if module_fqn:
                        module_symbol_counts[module_fqn] += 1
                        module_symbols[module_fqn].append({
                            "name": symbol_name,
                            "type": symbol_type
                        })
        
        # Check usage frequency for top symbols
        usage_scores = defaultdict(float)
        
        for module, symbols in module_symbols.items():
            if len(symbols) >= min_connections or module in module_scores:
                # Sample a few key symbols from each module
                key_symbols = symbols[:5]  # Check top 5 symbols
                
                for symbol in key_symbols:
                    try:
                        usages = get_symbol_usages(
                            repo=repo_path,
                            symbol_name_or_substring=symbol["name"],
                            working_directory=working_directory,
                            symbol_type=symbol["type"]
                        )
                        
                        # Count unique files using this symbol
                        unique_files = set()
                        for usage_line in usages:
                            if '|' in usage_line and not usage_line.startswith('-'):
                                parts = re.split(r'\s{2,}', usage_line.strip())
                                if len(parts) >= 3:
                                    unique_files.add(parts[2])
                        
                        usage_scores[module] += len(unique_files)
                    except:
                        pass
        
        # Calculate import connections from the graph
        import_graph = coderank_results["import_graph"]
        connection_counts = {}
        for module in module_scores:
            in_degree = import_graph.in_degree(module)
            out_degree = import_graph.out_degree(module)
            connection_counts[module] = in_degree + out_degree
        
        # Combine scores
        hotspot_scores = {}
        for module in set(list(module_scores.keys()) + list(usage_scores.keys())):
            coderank_score = module_scores.get(module, 0)
            usage_score = usage_scores.get(module, 0)
            symbol_count = module_symbol_counts.get(module, 0)
            connections = connection_counts.get(module, 0)
            
            # Weighted combination
            combined_score = (
                coderank_score * 100 +      # CodeRank is most important
                usage_score * 10 +          # Usage frequency
                symbol_count * 0.5 +        # Symbol density
                connections * 2             # Connection count
            )
            
            if connections >= min_connections or coderank_score > 0:
                hotspot_scores[module] = {
                    "combined_score": combined_score,
                    "coderank_score": coderank_score,
                    "usage_score": usage_score,
                    "symbol_count": symbol_count,
                    "connections": connections
                }
        
        # Sort by combined score
        sorted_hotspots = sorted(
            hotspot_scores.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )[:top_n]
        
        # Format output
        output_lines = ["=== Code Hotspot Analysis ===\n"]
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Minimum connections: {min_connections}")
        output_lines.append(f"Include external: {include_external}\n")
        output_lines.append("Top Code Hotspots (modules critical to the codebase):\n")
        
        output_lines.append(f"{'Module'.ljust(40)} | {'Hotspot Score'.rjust(13)} | {'CodeRank'.rjust(9)} | {'Usage'.rjust(6)} | {'Symbols'.rjust(8)} | {'Links'.rjust(6)}")
        output_lines.append("-" * 95)
        
        for module, scores in sorted_hotspots:
            output_lines.append(
                f"{module[:40].ljust(40)} | "
                f"{scores['combined_score']:13.2f} | "
                f"{scores['coderank_score']:9.4f} | "
                f"{scores['usage_score']:6.0f} | "
                f"{scores['symbol_count']:8d} | "
                f"{scores['connections']:6d}"
            )
        
        # Add insights
        output_lines.append("\n=== Insights ===")
        if sorted_hotspots:
            top_module = sorted_hotspots[0][0]
            output_lines.append(f"• Most critical module: {top_module}")
            
            high_usage = [m for m, s in sorted_hotspots if s['usage_score'] > 10]
            if high_usage:
                output_lines.append(f"• Highly used modules: {', '.join(high_usage[:3])}")
            
            high_complexity = [m for m, s in sorted_hotspots if s['symbol_count'] > 50]
            if high_complexity:
                output_lines.append(f"• Complex modules (many symbols): {', '.join(high_complexity[:3])}")
            
            highly_connected = [m for m, s in sorted_hotspots if s['connections'] > 20]
            if highly_connected:
                output_lines.append(f"• Highly connected modules: {', '.join(highly_connected[:3])}")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in find_code_hotspots: {str(e)}"


@mcp.tool()
def trace_dependency_impact(
    repo_path: str,
    target_module: str,
    working_directory: str,
    analysis_type: str = "dependency",
    max_depth: int = 3,
    change_type: str = "modify"
) -> str:
    """
    Trace dependency chains and analyze refactoring impact for a module.
    Combines dependency tracing with impact analysis.
    
    Args:
        repo_path: Repository path
        target_module: Module to analyze (e.g., 'src.auth.middleware')
        working_directory: Working directory (absolute path)
        analysis_type: "dependency" for chain tracing, "refactoring" for impact analysis, "both" for combined
        max_depth: Maximum depth to trace dependencies
        change_type: For refactoring - "modify", "split", "merge", or "remove"
    
    Returns:
        Dependency chains and/or refactoring impact analysis
    """
    try:
        output_lines = []
        
        # Get CodeRank data
        coderank_results = calculate_coderank(repo_path=repo_path)
        module_scores = coderank_results["module_ranks"]
        module_map = coderank_results["module_map"]
        import_graph = coderank_results["import_graph"]
        
        target_score = module_scores.get(target_module, 0)
        
        if target_module not in module_scores:
            return f"Module '{target_module}' not found in repository. Available modules: {', '.join(list(module_scores.keys())[:10])}..."
        
        if analysis_type in ["dependency", "both"]:
            output_lines.append("=== Dependency Chain Analysis ===\n")
            output_lines.append(f"Target Module: {target_module}")
            output_lines.append(f"CodeRank Score: {target_score:.4f}")
            output_lines.append(f"Max Depth: {max_depth}\n")
            
            # Get direct dependencies from the import graph
            upstream_deps = set()
            downstream_deps = set()
            
            # Upstream: what this module imports
            for edge in import_graph.out_edges(target_module):
                upstream_deps.add(edge[1])
            
            # Downstream: what imports this module
            for edge in import_graph.in_edges(target_module):
                downstream_deps.add(edge[0])
            
            output_lines.append("Upstream Dependencies (what this module imports):")
            if upstream_deps:
                for dep in sorted(upstream_deps):
                    score = module_scores.get(dep, 0)
                    output_lines.append(f"  • {dep} (score: {score:.4f})")
            else:
                output_lines.append("  • None")
            
            output_lines.append(f"\nDownstream Dependencies (what imports this module):")
            if downstream_deps:
                for dep in sorted(downstream_deps, key=lambda x: module_scores.get(x, 0), reverse=True):
                    score = module_scores.get(dep, 0)
                    output_lines.append(f"  • {dep} (score: {score:.4f})")
            else:
                output_lines.append("  • None")
            
            # Trace deeper dependencies if requested
            if max_depth > 1:
                output_lines.append(f"\n=== Extended Dependency Chain (up to {max_depth} levels) ===")
                
                # BFS for dependency chains
                visited = {target_module}
                current_level = {target_module}
                
                for level in range(1, max_depth + 1):
                    next_level = set()
                    level_deps = defaultdict(list)
                    
                    for module in current_level:
                        # Get dependencies
                        for edge in import_graph.out_edges(module):
                            if edge[1] not in visited:
                                next_level.add(edge[1])
                                level_deps[f"Level {level} upstream"].append(edge[1])
                        
                        for edge in import_graph.in_edges(module):
                            if edge[0] not in visited:
                                next_level.add(edge[0])
                                level_deps[f"Level {level} downstream"].append(edge[0])
                    
                    if level_deps:
                        for dep_type, deps in level_deps.items():
                            output_lines.append(f"\n{dep_type}:")
                            for dep in sorted(deps, key=lambda x: module_scores.get(x, 0), reverse=True)[:10]:
                                score = module_scores.get(dep, 0)
                                output_lines.append(f"  • {dep} (score: {score:.4f})")
                    
                    visited.update(next_level)
                    current_level = next_level
                    
                    if not next_level:
                        break
            
            # Calculate dependency metrics
            output_lines.append(f"\nDependency Metrics:")
            output_lines.append(f"  • Direct upstream dependencies: {len(upstream_deps)}")
            output_lines.append(f"  • Direct downstream dependencies: {len(downstream_deps)}")
            output_lines.append(f"  • Total direct connections: {len(upstream_deps) + len(downstream_deps)}")
            output_lines.append(f"  • Import centrality: {import_graph.degree(target_module)}")
        
        if analysis_type in ["refactoring", "both"]:
            if analysis_type == "both":
                output_lines.append("\n" + "="*50 + "\n")
            
            output_lines.append("=== Refactoring Impact Analysis ===\n")
            output_lines.append(f"Target Module: {target_module}")
            output_lines.append(f"Change Type: {change_type}")
            output_lines.append(f"Module Importance Score: {target_score:.4f}\n")
            
            # Get module file path
            module_file = module_map.get(target_module)
            
            # Get all symbols in the target module
            symbols = []
            if module_file:
                symbols_output = get_repo_symbols(
                    repo=repo_path,
                    working_directory=working_directory,
                    file_must_contain=os.path.relpath(module_file, repo_path)
                )
                symbols = [line for line in symbols_output if '|' in line and not line.startswith('-') and 'Symbol' not in line]
            
            symbol_count = len(symbols)
            
            # Find usages of symbols from this module
            total_usages = 0
            affected_files = set()
            
            for symbol_line in symbols[:10]:  # Sample first 10 symbols
                parts = re.split(r'\s{2,}', symbol_line.strip())
                if len(parts) >= 1:
                    symbol_name = parts[0]
                    try:
                        usages = get_symbol_usages(
                            repo=repo_path,
                            symbol_name_or_substring=symbol_name,
                            working_directory=working_directory
                        )
                        for usage_line in usages:
                            if '|' in usage_line and not usage_line.startswith('-'):
                                usage_parts = re.split(r'\s{2,}', usage_line.strip())
                                if len(usage_parts) >= 3:
                                    affected_files.add(usage_parts[2])
                                    total_usages += 1
                    except:
                        pass
            
            # Get downstream dependencies from earlier analysis or recalculate
            downstream_deps = set()
            for edge in import_graph.in_edges(target_module):
                downstream_deps.add(edge[0])
            
            # Impact assessment based on change type
            output_lines.append("Impact Assessment:")
            
            if change_type == "remove":
                output_lines.append(f"  • CRITICAL: Removing this module will break {len(downstream_deps)} dependent modules")
                output_lines.append(f"  • Estimated {total_usages * (symbol_count / 10)} symbol usages will need to be updated")
                output_lines.append(f"  • At least {len(affected_files)} files will be directly affected")
                
            elif change_type == "split":
                output_lines.append(f"  • Splitting will require updating imports in {len(downstream_deps)} modules")
                output_lines.append(f"  • {symbol_count} symbols need to be redistributed")
                output_lines.append(f"  • Consider grouping by: functionality, dependencies, or usage patterns")
                
            elif change_type == "merge":
                output_lines.append(f"  • Merging will consolidate {symbol_count} symbols")
                output_lines.append(f"  • May increase module complexity")
                connections = import_graph.degree(target_module)
                output_lines.append(f"  • Current module connections: {connections}")
                
            else:  # modify
                output_lines.append(f"  • Modifications will affect {len(downstream_deps)} dependent modules")
                output_lines.append(f"  • Estimated {total_usages * (symbol_count / 10)} symbol usages may need review")
                output_lines.append(f"  • Test coverage recommended for {len(affected_files)} files")

            
            if len(downstream_deps) > 0:
                output_lines.append(f"\n  • Update imports in these high-priority modules first:")
                priority_deps = sorted(
                    [(dep, module_scores.get(dep, 0)) for dep in downstream_deps],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for dep, score in priority_deps:
                    output_lines.append(f"    - {dep} (score: {score:.4f})")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in trace_dependency_impact: {str(e)}"


@mcp.tool()
def smart_code_search(
    keyword: str,
    repo_path: str,
    working_directory: str,
    rank_results: bool = True,
    context_lines: int = 3,
    max_results: int = 20
) -> str:
    """
    Enhanced search that combines ripgrep with CodeRank to prioritize results
    from more important modules.
    
    Args:
        keyword: Search term (supports regex)
        repo_path: Repository path
        working_directory: Working directory (absolute path)
        rank_results: Sort results by module importance
        context_lines: Lines of context around matches
        max_results: Maximum number of results to return
    
    Returns:
        Search results prioritized by code importance
    """
    try:
        # Get CodeRank scores if ranking is enabled
        module_scores = {}
        if rank_results:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
        
        # Run ripgrep search
        rg_result = subprocess.run(
            [
                "rg", "-n", "-i",
                "-B", str(context_lines), "-A", str(context_lines),
                "--json",  # Use JSON output for easier parsing
                keyword, repo_path
            ],
            capture_output=True,
            text=True,
            cwd=working_directory
        )
        
        if rg_result.returncode != 0:
            # Try without JSON for better error message
            simple_result = subprocess.run(
                ["rg", "-n", "-i", keyword, repo_path],
                capture_output=True,
                text=True,
                cwd=working_directory
            )
            if simple_result.returncode != 0:
                return f"No results found for keyword: {keyword}"
        
        # Parse ripgrep JSON output
        matches = []
        current_match = None
        
        for line in rg_result.stdout.split('\n'):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if data.get('type') == 'match':
                    file_path = data['data']['path']['text']
                    line_number = data['data']['line_number']
                    line_content = data['data']['lines']['text'].rstrip()
                    
                    # Convert file path to module name
                    module_name = path_to_module_fqn(file_path, repo_path)
                    if not module_name:
                        # For non-Python files or files that couldn't be converted
                        module_name = os.path.relpath(file_path, repo_path)
                    
                    # Get module score
                    score = module_scores.get(module_name, 0) if rank_results else 0
                    
                    if current_match and current_match['file'] == file_path and \
                       abs(current_match['line_number'] - line_number) <= context_lines + 1:
                        # Add to existing match context
                        current_match['context_lines'].append({
                            'line_number': line_number,
                            'content': line_content,
                            'is_match': True
                        })
                        current_match['match_count'] += 1
                    else:
                        # Save previous match and start new one
                        if current_match:
                            matches.append(current_match)
                        
                        current_match = {
                            'file': file_path,
                            'module': module_name,
                            'score': score,
                            'line_number': line_number,
                            'match_count': 1,
                            'context_lines': [{
                                'line_number': line_number,
                                'content': line_content,
                                'is_match': True
                            }]
                        }
                
                elif data.get('type') == 'context' and current_match:
                    line_number = data['data']['line_number']
                    line_content = data['data']['lines']['text'].rstrip()
                    current_match['context_lines'].append({
                        'line_number': line_number,
                        'content': line_content,
                        'is_match': False
                    })
            except json.JSONDecodeError:
                continue
        
        # Don't forget the last match
        if current_match:
            matches.append(current_match)
        
        # Sort matches by score if ranking is enabled
        if rank_results:
            matches.sort(key=lambda x: (x['score'], x['match_count']), reverse=True)
        
        # Limit results
        matches = matches[:max_results]
        
        # Format output
        output_lines = ["=== Smart Code Search Results ===\n"]
        output_lines.append(f"Search term: '{keyword}'")
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Ranking: {'Enabled (by module importance)' if rank_results else 'Disabled'}")
        output_lines.append(f"Found {len(matches)} matches (showing up to {max_results})\n")
        
        for i, match in enumerate(matches, 1):
            output_lines.append(f"--- Match {i}/{len(matches)} ---")
            output_lines.append(f"File: {os.path.relpath(match['file'], repo_path)}")
            output_lines.append(f"Module: {match['module']}")
            if rank_results and match['score'] > 0:
                output_lines.append(f"Importance Score: {match['score']:.4f}")
            output_lines.append(f"Matches in context: {match['match_count']}")
            output_lines.append("")
            
            # Sort context lines by line number
            context = sorted(match['context_lines'], key=lambda x: x['line_number'])

            for ctx in context:
                line_marker = ">>>" if ctx['is_match'] else "   "
                output_lines.append(f"{line_marker} {ctx['line_number']:5d}: {ctx['content']}")
            
            output_lines.append("")  # Empty line between matches
        
        # Add summary if ranking was used
        if rank_results and matches:
            output_lines.append("\n=== Search Insights ===")
            
            # Find high-importance matches
            high_importance = [m for m in matches if m['score'] > 0.01]
            if high_importance:
                output_lines.append(f"• Found {len(high_importance)} matches in high-importance modules")
                top_modules = list(set(m['module'] for m in high_importance[:5]))
                output_lines.append(f"• Top modules with matches: {', '.join(top_modules)}")
            
            # Find files with multiple matches
            file_match_counts = defaultdict(int)
            for m in matches:
                file_match_counts[m['file']] += m['match_count']
            
            multi_match_files = [(f, c) for f, c in file_match_counts.items() if c > 3]
            if multi_match_files:
                multi_match_files.sort(key=lambda x: x[1], reverse=True)
                output_lines.append(f"\n• Files with multiple matches:")
                for file, count in multi_match_files[:3]:
                    rel_path = os.path.relpath(file, repo_path)
                    output_lines.append(f"  - {rel_path}: {count} matches")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in smart_code_search: {str(e)}"


# ===============================================
# Recent Changes Analysis Tools
# ===============================================

@mcp.tool()
def analyze_recent_changes(
    repo_path: str,
    days_back: int = 30,
    target_branch: str = "main",
    min_commits: int = 2,
    top_n: int = 20,
    include_stats: bool = True
) -> str:
    """
    Analyze recent changes using CodeRank to identify most important modifications.
    Aggregates changes over the last N days and ranks them by impact.
    
    Args:
        repo_path: Repository path (absolute)
        days_back: Number of days to look back for commits
        target_branch: Branch to analyze (default: main)
        min_commits: Minimum commits to a file to be considered
        top_n: Number of top changes to return
        include_stats: Include detailed statistics
    
    Returns:
        Ranked list of most important recent changes with metrics
    """
    try:
        # First, get CodeRank scores for context
        coderank_results = calculate_coderank(repo_path=repo_path)
        module_scores = coderank_results["module_ranks"]
        module_map = coderank_results["module_map"]
        
        # Get recent commits
        since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Get commit data with file changes
        git_log_cmd = [
            "git", "-C", repo_path, "log",
            f"--since={since_date}",
            f"{target_branch}",
            "--name-status",
            "--format=%H|%ae|%an|%ad|%s",
            "--date=short"
        ]
        
        log_result = subprocess.run(git_log_cmd, capture_output=True, text=True)
        if log_result.returncode != 0:
            return f"Error getting git log: {log_result.stderr}"
        
        # Parse commits and changes
        commits_data = []
        current_commit = None
        
        for line in log_result.stdout.split('\n'):
            if not line.strip():
                continue
                
            if '|' in line and not line.startswith(('A\t', 'M\t', 'D\t')):
                # This is a commit line
                parts = line.split('|', 4)
                if len(parts) >= 5:
                    if current_commit:
                        commits_data.append(current_commit)
                    
                    current_commit = {
                        'hash': parts[0],
                        'author_email': parts[1],
                        'author_name': parts[2],
                        'date': parts[3],
                        'message': parts[4],
                        'files': []
                    }
            elif line.startswith(('A\t', 'M\t', 'D\t')) and current_commit:
                # This is a file change
                status, filepath = line.split('\t', 1)
                if filepath.endswith('.py'):
                    current_commit['files'].append({
                        'status': status,
                        'path': filepath
                    })
        
        if current_commit:
            commits_data.append(current_commit)
        
        # Aggregate changes by module
        module_changes: Dict[str, ModuleChangeData] = defaultdict(create_module_change_data)
        
        # Process each commit
        for commit in commits_data:
            for file_change in commit['files']:
                filepath = file_change['path']
                abs_filepath = os.path.join(repo_path, filepath)
                
                # Convert to module name
                module_fqn = path_to_module_fqn(abs_filepath, repo_path)
                if not module_fqn:
                    continue
                
                # Update module change data
                change_data = module_changes[module_fqn]
                change_data['commits'].append(commit['hash'][:7])
                change_data['authors'].add(commit['author_name'])
                change_data['commit_count'] += 1
                change_data['files'].add(filepath)
                change_data['coderank_score'] = module_scores.get(module_fqn, 0)
        
        # Get line change statistics if requested
        if include_stats:
            for module_fqn, data in module_changes.items():
                if data['commit_count'] >= min_commits:
                    # Get diff stats for this module's files
                    for filepath in data['files']:
                        diff_cmd = [
                            "git", "-C", repo_path, "diff",
                            f"--since={since_date}",
                            f"{target_branch}",
                            "--numstat",
                            "--", filepath
                        ]
                        diff_result = subprocess.run(diff_cmd, capture_output=True, text=True)
                        
                        if diff_result.returncode == 0 and diff_result.stdout:
                            lines = diff_result.stdout.strip().split('\n')
                            for line in lines:
                                if line:
                                    parts = line.split('\t')
                                    if len(parts) >= 2:
                                        try:
                                            added = int(parts[0]) if parts[0] != '-' else 0
                                            deleted = int(parts[1]) if parts[1] != '-' else 0
                                            data['lines_changed'] += added + deleted
                                        except ValueError:
                                            pass
        
        # Calculate change impact scores
        change_scores = []
        
        for module_fqn, data in module_changes.items():
            if data['commit_count'] < min_commits:
                continue
            
            # Calculate composite score
            impact_score = (
                data['coderank_score'] * 1000 +  # Module importance (heavily weighted)
                data['commit_count'] * 10 +       # Frequency of changes
                len(data['authors']) * 50 +       # Contributor diversity
                data['lines_changed'] * 0.1       # Size of changes
            )
            
            # Get file path for this module
            module_file_path = module_map.get(module_fqn, "N/A")
            rel_file_path = os.path.relpath(module_file_path, repo_path) if module_file_path != "N/A" else "N/A"
            
            change_scores.append({
                'module': module_fqn,
                'file_path': rel_file_path,
                'impact_score': impact_score,
                'coderank_score': data['coderank_score'],
                'commit_count': data['commit_count'],
                'unique_authors': len(data['authors']),
                'lines_changed': data['lines_changed'],
                'authors': list(data['authors']),
                'recent_commits': data['commits'][-5:]  # Last 5 commit hashes
            })
        
        # Sort by impact score
        change_scores.sort(key=lambda x: x['impact_score'], reverse=True)
        top_changes = change_scores[:top_n]
        
        # Format output
        output_lines = ["=== Recent Changes Analysis (CodeRank-based) ===\n"]
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Analysis period: Last {days_back} days")
        output_lines.append(f"Target branch: {target_branch}")
        output_lines.append(f"Total commits analyzed: {len(commits_data)}")
        output_lines.append(f"Modules with significant changes: {len(change_scores)}\n")
        
        output_lines.append("Top Changed Modules by Impact:")
        output_lines.append(f"{'Module'.ljust(35)} | {'File Path'.ljust(35)} | {'Impact'.rjust(7)} | {'CodeRank'.rjust(8)} | {'Commits'.rjust(7)} | {'Authors'.rjust(7)} | {'Lines ±'.rjust(8)}")
        output_lines.append("-" * 130)
        
        for change in top_changes:
            output_lines.append(
                f"{change['module'][:35].ljust(35)} | "
                f"{change['file_path'][:35].ljust(35)} | "
                f"{change['impact_score']:7.1f} | "
                f"{change['coderank_score']:8.4f} | "
                f"{change['commit_count']:7d} | "
                f"{change['unique_authors']:7d} | "
                f"{change['lines_changed']:8d}"
            )
        
        # Add insights
        output_lines.append("\n=== Key Insights ===")
        
        # Find high-impact changes
        high_impact = [c for c in top_changes if c['coderank_score'] > 0.01]
        if high_impact:
            output_lines.append(f"\n• Critical Module Changes ({len(high_impact)} modules):")
            for change in high_impact[:5]:
                output_lines.append(f"  - {change['module']}: {change['commit_count']} commits by {change['unique_authors']} authors")
                output_lines.append(f"    File: {change['file_path']}")
        
        # Find hotspots (many commits)
        hotspots = [c for c in top_changes if c['commit_count'] > 10]
        if hotspots:
            output_lines.append(f"\n• Change Hotspots (frequently modified):")
            for change in hotspots[:3]:
                output_lines.append(f"  - {change['module']}: {change['commit_count']} commits")
                output_lines.append(f"    File: {change['file_path']}")
        
        # Find collaborative changes
        collaborative = [c for c in top_changes if c['unique_authors'] > 3]
        if collaborative:
            output_lines.append(f"\n• Collaborative Development Areas:")
            for change in collaborative[:3]:
                authors = ", ".join(change['authors'][:3])
                if len(change['authors']) > 3:
                    authors += f" +{len(change['authors'])-3} more"
                output_lines.append(f"  - {change['module']}: {authors}")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in analyze_recent_changes: {str(e)}"


@mcp.tool()
def get_commit_hotspots(
    repo_path: str,
    days_back: int = 30,
    min_cochange_frequency: int = 3,
    top_n: int = 10
) -> str:
    """
    Find modules that are frequently changed together in commits.
    Identifies coupled modules that might have hidden dependencies.
    
    Args:
        repo_path: Repository path
        days_back: Days to analyze
        min_cochange_frequency: Minimum times modules must change together
        top_n: Number of top coupled module pairs to return
    
    Returns:
        Analysis of modules that frequently change together
    """
    try:
        # Get CodeRank data for context
        coderank_results = calculate_coderank(repo_path=repo_path)
        module_scores = coderank_results["module_ranks"]
        
        since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Get commits with changed files
        git_cmd = [
            "git", "-C", repo_path, "log",
            f"--since={since_date}",
            "--name-only",
            "--format=%H|%an|%ae|%ad",
            "--date=short"
        ]
        
        result = subprocess.run(git_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error getting git log: {result.stderr}"
        
        # Parse commits and their files
        commits = []
        current_commit = None
        
        for line in result.stdout.split('\n'):
            if '|' in line:
                if current_commit and current_commit['modules']:
                    commits.append(current_commit)
                
                parts = line.split('|')
                current_commit = {
                    'hash': parts[0],
                    'author': parts[1],
                    'email': parts[2],
                    'date': parts[3],
                    'modules': set()
                }
            elif line.strip() and line.endswith('.py') and current_commit:
                # Convert file to module
                abs_path = os.path.join(repo_path, line.strip())
                module_fqn = path_to_module_fqn(abs_path, repo_path)
                if module_fqn:
                    current_commit['modules'].add(module_fqn)
        
        if current_commit and current_commit['modules']:
            commits.append(current_commit)
        
        # Find co-occurring modules
        cochange_pairs: Dict[Tuple[str, str], CochangeData] = defaultdict(create_cochange_data)
        
        for commit in commits:
            modules = list(commit['modules'])
            if len(modules) < 2:
                continue
            
            # Generate all pairs
            for i in range(len(modules)):
                for j in range(i + 1, len(modules)):
                    pair = tuple(sorted([modules[i], modules[j]]))
                    data = cochange_pairs[pair]
                    data['count'] += 1
                    data['commits'].append(commit['hash'][:7])
                    data['authors'].add(commit['author'])
                    
                    # Calculate combined CodeRank score
                    score1 = module_scores.get(modules[i], 0)
                    score2 = module_scores.get(modules[j], 0)
                    data['combined_coderank'] = score1 + score2
        
        # Filter and sort
        significant_pairs = []
        for pair, data in cochange_pairs.items():
            if data['count'] >= min_cochange_frequency:
                significant_pairs.append({
                    'modules': pair,
                    'frequency': data['count'],
                    'unique_authors': len(data['authors']),
                    'combined_coderank': data['combined_coderank'],
                    'coupling_score': data['count'] * data['combined_coderank'] * 100,
                    'recent_commits': data['commits'][-3:]
                })
        
        significant_pairs.sort(key=lambda x: x['coupling_score'], reverse=True)
        top_pairs = significant_pairs[:top_n]
        
        # Format output
        output_lines = ["=== Commit Hotspot Analysis ===\n"]
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Period: Last {days_back} days")
        output_lines.append(f"Total commits analyzed: {len(commits)}")
        output_lines.append(f"Coupled module pairs found: {len(significant_pairs)}\n")
        
        output_lines.append("Top Coupled Modules (frequently changed together):")
        output_lines.append(f"{'Module 1'.ljust(30)} | {'Module 2'.ljust(30)} | {'Freq'.rjust(4)} | {'Score'.rjust(7)}")
        output_lines.append("-" * 80)
        
        for pair_data in top_pairs:
            mod1, mod2 = pair_data['modules']
            output_lines.append(
                f"{mod1[:30].ljust(30)} | "
                f"{mod2[:30].ljust(30)} | "
                f"{pair_data['frequency']:4d} | "
                f"{pair_data['coupling_score']:7.1f}"
            )
        
        # Add insights
        output_lines.append("\n=== Coupling Insights ===")
        
        # Find high-importance coupled modules
        high_importance_pairs = [p for p in top_pairs if p['combined_coderank'] > 0.02]
        if high_importance_pairs:
            output_lines.append("\n• High-importance coupled modules:")
            for pair in high_importance_pairs[:3]:
                output_lines.append(f"  - {pair['modules'][0]} ↔ {pair['modules'][1]}")
                output_lines.append(f"    Changed together {pair['frequency']} times")
        
        # Find potential refactoring candidates
        high_coupling = [p for p in top_pairs if p['frequency'] > 10]
        if high_coupling:
            output_lines.append("\n• Potential refactoring candidates (high coupling):")
            for pair in high_coupling[:3]:
                output_lines.append(f"  - {pair['modules'][0]} & {pair['modules'][1]}")
                output_lines.append(f"    Consider: merge, extract common interface, or clarify boundaries")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in get_commit_hotspots: {str(e)}"


@mcp.tool()
def contributor_impact_analysis(
    repo_path: str,
    days_back: int = 90,
    min_commits: int = 5,
    focus_on_important_modules: bool = True
) -> str:
    """
    Analyze contributor patterns and their impact on important modules.
    Identifies key contributors and their areas of expertise.
    
    Args:
        repo_path: Repository path
        days_back: Days to analyze
        min_commits: Minimum commits by a contributor to be included
        focus_on_important_modules: Weight contributions by module importance
    
    Returns:
        Contributor impact analysis with expertise areas
    """
    try:
        # Get CodeRank data
        coderank_results = calculate_coderank(repo_path=repo_path)
        module_scores = coderank_results["module_ranks"]
        
        since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Get detailed commit data
        git_cmd = [
            "git", "-C", repo_path, "log",
            f"--since={since_date}",
            "--name-status",
            "--format=%H|%ae|%an|%ad|%s",
            "--date=short"
        ]
        
        result = subprocess.run(git_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error getting git log: {result.stderr}"
        
        # Parse contributor data
        contributor_data: Dict[str, ContributorData] = defaultdict(create_contributor_data)
        
        current_commit = None
        for line in result.stdout.split('\n'):
            if '|' in line and not line.startswith(('A\t', 'M\t', 'D\t')):
                parts = line.split('|', 4)
                if len(parts) >= 5:
                    current_commit = {
                        'hash': parts[0],
                        'email': parts[1],
                        'name': parts[2],
                        'date': parts[3],
                        'message': parts[4],
                        'modules': set()
                    }
            elif line.startswith(('A\t', 'M\t', 'D\t')) and current_commit:
                status, filepath = line.split('\t', 1)
                if filepath.endswith('.py'):
                    abs_path = os.path.join(repo_path, filepath)
                    module_fqn = path_to_module_fqn(abs_path, repo_path)
                    
                    if module_fqn:
                        contributor = contributor_data[current_commit['name']]
                        contributor['commits'] += 1
                        contributor['modules_touched'].add(module_fqn)
                        contributor['recent_commits'].append(current_commit['hash'][:7])
                        
                        # Track important module contributions
                        module_score = module_scores.get(module_fqn, 0)
                        if module_score > 0.01:
                            contributor['important_module_commits'] += 1
                        
                        if focus_on_important_modules:
                            contributor['impact_score'] += module_score * 100
        
        # Get line statistics for top contributors
        active_contributors = {name: data for name, data in contributor_data.items() 
                              if data['commits'] >= min_commits}
        
        for name, data in active_contributors.items():
            # Get contributor's line changes
            stat_cmd = [
                "git", "-C", repo_path, "log",
                f"--author={name}",
                f"--since={since_date}",
                "--pretty=tformat:",
                "--numstat"
            ]
            
            stat_result = subprocess.run(stat_cmd, capture_output=True, text=True)
            if stat_result.returncode == 0:
                for line in stat_result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 3 and parts[2].endswith('.py'):
                            try:
                                added = int(parts[0]) if parts[0] != '-' else 0
                                removed = int(parts[1]) if parts[1] != '-' else 0
                                data['lines_added'] += added
                                data['lines_removed'] += removed
                            except ValueError:
                                pass
        
        # Calculate final impact scores
        contributor_scores = []
        for name, data in active_contributors.items():
            if not focus_on_important_modules:
                # Alternative scoring without module importance
                data['impact_score'] = (
                    data['commits'] * 10 +
                    len(data['modules_touched']) * 5 +
                    (data['lines_added'] + data['lines_removed']) * 0.01
                )
            
            # Find expertise areas (most touched modules)
            module_touches = defaultdict(int)
            for module in data['modules_touched']:
                module_touches[module] += 1
            
            expertise_areas = sorted(
                [(m, c) for m, c in module_touches.items()],
                key=lambda x: (x[1], module_scores.get(x[0], 0)),
                reverse=True
            )[:3]
            
            contributor_scores.append({
                'name': name,
                'impact_score': data['impact_score'],
                'commits': data['commits'],
                'modules_touched': len(data['modules_touched']),
                'lines_changed': data['lines_added'] + data['lines_removed'],
                'important_module_commits': data['important_module_commits'],
                'expertise_areas': expertise_areas
            })
        
        # Sort by impact
        contributor_scores.sort(key=lambda x: x['impact_score'], reverse=True)
        
        # Format output
        output_lines = ["=== Contributor Impact Analysis ===\n"]
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Period: Last {days_back} days")
        output_lines.append(f"Active contributors: {len(contributor_scores)}")
        output_lines.append(f"Module importance weighting: {'Enabled' if focus_on_important_modules else 'Disabled'}\n")
        
        output_lines.append("Top Contributors by Impact:")
        output_lines.append(f"{'Contributor'.ljust(25)} | {'Impact'.rjust(7)} | {'Commits'.rjust(7)} | {'Modules'.rjust(7)} | {'Lines ±'.rjust(8)} | {'Critical'.rjust(8)}")
        output_lines.append("-" * 85)
        
        for contributor in contributor_scores[:15]:
            output_lines.append(
                f"{contributor['name'][:25].ljust(25)} | "
                f"{contributor['impact_score']:7.1f} | "
                f"{contributor['commits']:7d} | "
                f"{contributor['modules_touched']:7d} | "
                f"{contributor['lines_changed']:8d} | "
                f"{contributor['important_module_commits']:8d}"
            )
        
        # Add expertise breakdown
        output_lines.append("\n=== Contributor Expertise Areas ===")
        
        for contributor in contributor_scores[:10]:
            if contributor['expertise_areas']:
                output_lines.append(f"\n{contributor['name']}:")
                for module, touch_count in contributor['expertise_areas']:
                    module_score = module_scores.get(module, 0)
                    importance = "HIGH" if module_score > 0.01 else "normal"
                    output_lines.append(f"  • {module} ({touch_count} commits, {importance} importance)")
        
        # Add insights
        output_lines.append("\n=== Key Insights ===")
        
        # Find domain experts
        critical_contributors = [c for c in contributor_scores 
                               if c['important_module_commits'] > 10]
        if critical_contributors:
            output_lines.append("\n• Critical module experts:")
            for contrib in critical_contributors[:3]:
                output_lines.append(f"  - {contrib['name']}: {contrib['important_module_commits']} commits to critical modules")
        
        # Find broad contributors
        broad_contributors = [c for c in contributor_scores 
                            if c['modules_touched'] > 20]
        if broad_contributors:
            output_lines.append("\n• Broad impact contributors:")
            for contrib in broad_contributors[:3]:
                output_lines.append(f"  - {contrib['name']}: touched {contrib['modules_touched']} different modules")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in contributor_impact_analysis: {str(e)}"


@mcp.tool()
def change_propagation_analysis(
    repo_path: str,
    changed_module: str,
    days_back: int = 90,
    include_test_impact: bool = True
) -> str:
    """
    Analyze how changes in one module historically propagate to others.
    Uses commit history to predict ripple effects of changes.
    
    Args:
        repo_path: Repository path
        changed_module: Module to analyze (e.g., 'src.auth.user')
        days_back: Days of history to analyze
        include_test_impact: Include analysis of test file changes
    
    Returns:
        Analysis of likely modules to be affected by changes
    """
    try:
        # Get CodeRank data
        coderank_results = calculate_coderank(repo_path=repo_path)
        module_scores = coderank_results["module_ranks"]
        import_graph = coderank_results["import_graph"]
        
        if changed_module not in module_scores:
            return f"Module '{changed_module}' not found. Available modules: {', '.join(list(module_scores.keys())[:10])}..."
        
        since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Find commits that touched the target module
        module_file = coderank_results["module_map"].get(changed_module)
        if not module_file:
            return f"Could not find file for module {changed_module}"
        
        rel_module_file = os.path.relpath(module_file, repo_path)
        
        # Get commits that changed this module
        git_cmd = [
            "git", "-C", repo_path, "log",
            f"--since={since_date}",
            "--format=%H",
            "--", rel_module_file
        ]
        
        result = subprocess.run(git_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error getting git log: {result.stderr}"
        
        target_commits = result.stdout.strip().split('\n')
        if not target_commits or not target_commits[0]:
            return f"No commits found for module {changed_module} in the last {days_back} days"
        
        # For each commit, find what else changed
        propagation_data: Dict[str, PropagationData] = defaultdict(create_propagation_data)
        
        for commit_hash in target_commits:
            # Get all files changed in this commit
            files_cmd = [
                "git", "-C", repo_path, "show",
                "--name-only",
                "--format=",
                commit_hash
            ]
            
            files_result = subprocess.run(files_cmd, capture_output=True, text=True)
            if files_result.returncode == 0:
                changed_files = files_result.stdout.strip().split('\n')
                
                for filepath in changed_files:
                    if filepath and filepath.endswith('.py') and filepath != rel_module_file:
                        abs_filepath = os.path.join(repo_path, filepath)
                        other_module = path_to_module_fqn(abs_filepath, repo_path)
                        
                        if other_module:
                            data = propagation_data[other_module]
                            data['co_change_count'] += 1
                            data['commits'].append(commit_hash[:7])
                            data['module_score'] = module_scores.get(other_module, 0)
                            
                            # Check if it's a test file
                            if 'test' in filepath.lower():
                                data['is_test'] = True
                            
                            # Check if there's an import relationship
                            if (other_module in import_graph.successors(changed_module) or
                                changed_module in import_graph.successors(other_module)):
                                data['is_import_related'] = True
        
        # Calculate propagation scores
        propagation_scores = []
        for module, data in propagation_data.items():
            # Skip tests if not requested
            if not include_test_impact and data['is_test']:
                continue
            
            # Calculate likelihood score
            propagation_score = (
                data['co_change_count'] * 10 +                    # Frequency
                (50 if data['is_import_related'] else 0) +        # Import relationship
                data['module_score'] * 100                        # Module importance
            )
            
            propagation_scores.append({
                'module': module,
                'score': propagation_score,
                'co_changes': data['co_change_count'],
                'probability': data['co_change_count'] / len(target_commits),
                'is_import_related': data['is_import_related'],
                'is_test': data['is_test'],
                'module_importance': data['module_score']
            })
        
        # Sort by score
        propagation_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Format output
        output_lines = ["=== Change Propagation Analysis ===\n"]
        output_lines.append(f"Target Module: {changed_module}")
        output_lines.append(f"Module Importance: {module_scores[changed_module]:.4f}")
        output_lines.append(f"Analysis Period: Last {days_back} days")
        output_lines.append(f"Commits analyzed: {len(target_commits)}")
        output_lines.append(f"Include test impact: {include_test_impact}\n")
        
        # Direct dependencies from import graph
        direct_importers = list(import_graph.predecessors(changed_module))
        direct_imports = list(import_graph.successors(changed_module))
        
        output_lines.append("Direct Dependencies:")
        output_lines.append(f"  • Imports from {changed_module}: {len(direct_importers)} modules")
        output_lines.append(f"  • {changed_module} imports: {len(direct_imports)} modules\n")
        
        output_lines.append("Likely Affected Modules (based on historical patterns):")
        output_lines.append(f"{'Module'.ljust(35)} | {'Likelihood'.rjust(10)} | {'Co-changes'.rjust(10)} | {'Import Link'} | {'Type'}")
        output_lines.append("-" * 85)
        
        for prop in propagation_scores[:20]:
            import_marker = "Yes" if prop['is_import_related'] else "No"
            module_type = "Test" if prop['is_test'] else "Code"
            
            output_lines.append(
                f"{prop['module'][:35].ljust(35)} | "
                f"{prop['probability']:10.1%} | "
                f"{prop['co_changes']:10d} | "
                f"{import_marker:11} | "
                f"{module_type}"
            )
        
        # Add insights
        output_lines.append("\n=== Propagation Insights ===")
        
        # High probability changes
        high_prob = [p for p in propagation_scores if p['probability'] > 0.5]
        if high_prob:
            output_lines.append(f"\n• High probability ripple effects ({len(high_prob)} modules):")
            for prop in high_prob[:5]:
                reason = "import dependency" if prop['is_import_related'] else "historical coupling"
                output_lines.append(f"  - {prop['module']} ({prop['probability']:.0%} chance, {reason})")
        
        # Test impact
        if include_test_impact:
            test_impacts = [p for p in propagation_scores if p['is_test']]
            if test_impacts:
                output_lines.append(f"\n• Test files likely to need updates ({len(test_impacts)} files):")
                for prop in test_impacts[:5]:
                    output_lines.append(f"  - {prop['module']} ({prop['probability']:.0%} historical correlation)")
        
        # Import-related changes
        import_related = [p for p in propagation_scores if p['is_import_related']]
        if import_related:
            output_lines.append(f"\n• Import-dependent modules ({len(import_related)} modules):")
            for prop in import_related[:5]:
                output_lines.append(f"  - {prop['module']} (direct import relationship)")
        
        # Risk assessment
        output_lines.append("\n=== Change Risk Assessment ===")
        output_lines.append(f"• Estimated modules affected: {len([p for p in propagation_scores if p['probability'] > 0.3])}")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in change_propagation_analysis: {str(e)}"


# ===============================================
# Codebase Understanding & Development Tools
# ===============================================

@mcp.tool()
def trace_data_flow(
    repo_path: str,
    working_directory: str,
    data_identifier: str,
    max_depth: int = 5,
    include_transformations: bool = True,
    show_side_effects: bool = True
) -> str:
    """
    Trace how specific data flows through the system from source to destination.
    
    Use this tool when you need to understand:
    - How a piece of data (user_id, email, order_data, etc.) moves through the codebase
    - What functions transform or modify the data
    - Where data comes from and where it goes
    - What side effects might occur when processing this data
    
    Perfect for debugging data-related issues, understanding data dependencies,
    or planning changes that affect data flow.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        data_identifier: Name of the data to trace (e.g., "user_id", "email", "order")
        max_depth: How deep to trace the data flow
        include_transformations: Whether to show data transformation points
        show_side_effects: Whether to identify potential side effects
    
    Returns:
        Comprehensive data flow analysis with transformation points and dependencies
    """
    try:
        output_lines = ["=== Data Flow Analysis ===\n"]
        output_lines.append(f"Tracing: {data_identifier}")
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Max depth: {max_depth}\n")
        
        # Use ripgrep to find all mentions of the data identifier
        rg_result = subprocess.run(
            ["rg", "-n", "-i", "--type", "py", data_identifier, repo_path],
            capture_output=True,
            text=True,
            cwd=working_directory
        )
        
        if rg_result.returncode != 0:
            return f"No occurrences found for data identifier: {data_identifier}"
        
        # Parse occurrences and categorize them
        occurrences = []
        for line in rg_result.stdout.split('\n'):
            if ':' in line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    file_path = parts[0]
                    line_number = parts[1]
                    content = parts[2].strip()
                    
                    # Categorize the occurrence
                    category = "usage"
                    if any(pattern in content.lower() for pattern in ['def ', 'class ', 'return ', '=']):
                        if 'def ' in content:
                            category = "function_parameter"
                        elif 'return ' in content:
                            category = "return_value"
                        elif '=' in content and data_identifier in content.split('=')[0]:
                            category = "assignment"
                        else:
                            category = "definition"
                    
                    occurrences.append({
                        'file': file_path,
                        'line': line_number,
                        'content': content,
                        'category': category
                    })
        
        # Group by category
        categories = {}
        for occ in occurrences:
            cat = occ['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(occ)
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
        except:
            module_scores = {}
        
        # Analyze each category
        for category, items in categories.items():
            output_lines.append(f"\n=== {category.replace('_', ' ').title()} ===")
            
            # Sort by importance if possible
            scored_items = []
            for item in items:
                module_fqn = path_to_module_fqn(os.path.join(repo_path, item['file']), repo_path)
                score = module_scores.get(module_fqn, 0) if module_fqn else 0
                scored_items.append((item, score))
            
            scored_items.sort(key=lambda x: x[1], reverse=True)
            
            for item, score in scored_items[:10]:  # Show top 10 per category
                rel_path = os.path.relpath(item['file'], repo_path)
                importance = f" (importance: {score:.4f})" if score > 0 else ""
                output_lines.append(f"  • {rel_path}:{item['line']}{importance}")
                output_lines.append(f"    {item['content']}")
        
        # If include_transformations, look for transformation patterns
        if include_transformations:
            output_lines.append(f"\n=== Data Transformations ===")
            transform_patterns = [
                f"{data_identifier}.transform",
                f"process_{data_identifier}",
                f"convert_{data_identifier}",
                f"format_{data_identifier}",
                f"validate_{data_identifier}"
            ]
            
            transformations_found = []
            for pattern in transform_patterns:
                transform_result = subprocess.run(
                    ["rg", "-n", "-i", "--type", "py", pattern, repo_path],
                    capture_output=True,
                    text=True,
                    cwd=working_directory
                )
                
                if transform_result.returncode == 0:
                    for line in transform_result.stdout.split('\n')[:5]:  # Top 5 matches
                        if ':' in line:
                            transformations_found.append(line)
            
            if transformations_found:
                for transform in transformations_found:
                    output_lines.append(f"  • {transform}")
            else:
                output_lines.append("  • No explicit transformation patterns found")
        
        # Note: Side effect analysis removed due to hardcoded patterns
        # Users should use discover_side_effects tool for detailed side effect analysis
        if show_side_effects:
            output_lines.append(f"\n=== Side Effects Analysis ===")
            output_lines.append("  • Use discover_side_effects tool for detailed side effect analysis")
            output_lines.append("  • Provide custom side effect patterns for your specific use case")
        
        # Summary
        output_lines.append(f"\n=== Summary ===")
        output_lines.append(f"Total occurrences found: {len(occurrences)}")
        output_lines.append(f"Categories: {', '.join(categories.keys())}")
        output_lines.append(f"Files involved: {len(set(occ['file'] for occ in occurrences))}")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in trace_data_flow: {str(e)}"


@mcp.tool()
def analyze_error_patterns(
    repo_path: str,
    working_directory: str,
    focus_area: Optional[str] = None,
    custom_patterns: Optional[Dict[str, str]] = None,
    custom_antipatterns: Optional[Dict[str, str]] = None,
    include_antipatterns: bool = True,
    show_evolution: bool = True,
    days_back: int = 180
) -> str:
    """
    Discover error handling patterns and inconsistencies in the codebase.
    
    Use this tool when you need to understand:
    - How errors are typically handled in this codebase
    - What error handling patterns to follow for consistency
    - Where error handling might be missing or inconsistent
    - How error handling has evolved over time
    
    Essential for implementing proper error handling that matches the codebase style,
    debugging error-related issues, or improving error handling consistency.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        focus_area: Specific area to focus on (e.g., "database", "api", "file_io")
        custom_patterns: Custom regex patterns for error handling (overrides defaults)
        custom_antipatterns: Custom regex patterns for antipatterns (overrides defaults)
        include_antipatterns: Whether to identify problematic error handling
        show_evolution: Whether to show how error handling has changed
        days_back: Days of git history to analyze for evolution
    
    Returns:
        Comprehensive error handling analysis with patterns and recommendations
    """
    try:
        output_lines = ["=== Error Handling Pattern Analysis ===\n"]
        output_lines.append(f"Repository: {repo_path}")
        if focus_area:
            output_lines.append(f"Focus area: {focus_area}")
        output_lines.append(f"Include antipatterns: {include_antipatterns}")
        output_lines.append(f"Show evolution: {show_evolution}\n")
        
        # Define error handling patterns to search for - use custom patterns if provided
        if custom_patterns:
            error_patterns = custom_patterns
        else:
            error_patterns = {
                "try_except": r"try:|except\s+\w+:",
                "raise_statements": r"raise\s+\w+",
                "error_returns": r"return.*[Ee]rror|return.*[Ff]alse",
                "logging_errors": r"log\.error|logger\.error|logging\.error",
                "custom_exceptions": r"class\s+\w*[Ee]rror|class\s+\w*[Ee]xception",
                "error_checking": r"if.*error|if.*failed|if.*success.*false"
            }
        
        # If focus_area specified, adjust search
        search_path = repo_path
        if focus_area:
            # Try to find focus area specific files
            focus_result = subprocess.run(
                ["rg", "-l", "-i", "--type", "py", focus_area, repo_path],
                capture_output=True,
                text=True,
                cwd=working_directory
            )
            if focus_result.returncode == 0:
                output_lines.append(f"Found {len(focus_result.stdout.split())} files related to {focus_area}")
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
        except:
            module_scores = {}
        
        pattern_results = {}
        
        # Search for each error handling pattern
        for pattern_name, pattern_regex in error_patterns.items():
            rg_result = subprocess.run(
                ["rg", "-n", "-i", "--type", "py", pattern_regex, search_path],
                capture_output=True,
                text=True,
                cwd=working_directory
            )
            
            if rg_result.returncode == 0:
                lines = rg_result.stdout.split('\n')
                pattern_results[pattern_name] = []
                
                for line in lines[:20]:  # Limit to top 20 per pattern
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path = parts[0]
                            line_number = parts[1]
                            content = parts[2].strip()
                            
                            # Get module importance
                            module_fqn = path_to_module_fqn(os.path.join(repo_path, file_path), repo_path)
                            score = module_scores.get(module_fqn, 0) if module_fqn else 0
                            
                            pattern_results[pattern_name].append({
                                'file': file_path,
                                'line': line_number,
                                'content': content,
                                'score': score
                            })
                
                # Sort by importance
                pattern_results[pattern_name].sort(key=lambda x: x['score'], reverse=True)
        
        # Analyze and present results
        for pattern_name, results in pattern_results.items():
            if results:
                output_lines.append(f"\n=== {pattern_name.replace('_', ' ').title()} ({len(results)} occurrences) ===")
                
                # Show top examples
                for result in results[:5]:
                    rel_path = os.path.relpath(result['file'], repo_path)
                    importance = f" (importance: {result['score']:.4f})" if result['score'] > 0 else ""
                    output_lines.append(f"  • {rel_path}:{result['line']}{importance}")
                    output_lines.append(f"    {result['content']}")
                
                if len(results) > 5:
                    output_lines.append(f"    ... and {len(results) - 5} more occurrences")
        
        # Look for antipatterns if requested
        if include_antipatterns:
            output_lines.append(f"\n=== Potential Error Handling Issues ===")
            
            if custom_antipatterns:
                antipatterns = custom_antipatterns
            else:
                antipatterns = {
                    "bare_except": r"except:",
                    "pass_in_except": r"except.*:\s*pass",
                    "print_errors": r"print.*error|print.*exception",
                    "swallowed_exceptions": r"except.*:\s*(?:pass|continue|return)"
                }
            
            issues_found = []
            for antipattern_name, antipattern_regex in antipatterns.items():
                ap_result = subprocess.run(
                    ["rg", "-n", "-i", "--type", "py", antipattern_regex, repo_path],
                    capture_output=True,
                    text=True,
                    cwd=working_directory
                )
                
                if ap_result.returncode == 0:
                    lines = ap_result.stdout.split('\n')[:5]  # Top 5 issues
                    for line in lines:
                        if line.strip():
                            issues_found.append(f"{antipattern_name}: {line}")
            
            if issues_found:
                for issue in issues_found:
                    output_lines.append(f"  ⚠️  {issue}")
            else:
                output_lines.append("  ✅ No obvious error handling antipatterns detected")
        
        # Show evolution if requested
        if show_evolution:
            output_lines.append(f"\n=== Error Handling Evolution (last {days_back} days) ===")
            
            since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Look for error-related commits
            git_cmd = [
                "git", "-C", repo_path, "log",
                f"--since={since_date}",
                "--grep=error",
                "--grep=exception",
                "--grep=fix",
                "--oneline",
                "--max-count=10"
            ]
            
            git_result = subprocess.run(git_cmd, capture_output=True, text=True)
            if git_result.returncode == 0 and git_result.stdout:
                commits = git_result.stdout.strip().split('\n')
                output_lines.append(f"Recent error-related commits ({len(commits)}):")
                for commit in commits:
                    output_lines.append(f"  • {commit}")
            else:
                output_lines.append("No recent error-related commits found")
        
        # Summary and recommendations
        output_lines.append(f"\n=== Error Handling Summary ===")
        total_patterns = sum(len(results) for results in pattern_results.values())
        output_lines.append(f"Total error handling patterns found: {total_patterns}")
        
        # Most common pattern
        if pattern_results:
            most_common = max(pattern_results.items(), key=lambda x: len(x[1]))
            output_lines.append(f"Most common pattern: {most_common[0]} ({len(most_common[1])} occurrences)")
        
        # Recommendations
        output_lines.append(f"\n=== Recommendations ===")
        if pattern_results.get("try_except"):
            output_lines.append("✅ Codebase uses try/except blocks - follow this pattern")
        if pattern_results.get("logging_errors"):
            output_lines.append("✅ Codebase logs errors - continue this practice")
        if pattern_results.get("custom_exceptions"):
            output_lines.append("✅ Custom exceptions defined - use appropriate ones")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in analyze_error_patterns: {str(e)}"


@mcp.tool()
def trace_feature_implementation(
    repo_path: str,
    working_directory: str,
    feature_keywords: List[str],
    file_categories: Dict[str, List[str]],
    include_tests: bool = True,
    include_config: bool = True,
    trace_depth: int = 3
) -> str:
    """
    Map all code involved in implementing a specific feature from UI to data layer.
    
    Use this tool when you need to:
    - Understand all components involved in a feature before modifying it
    - Map feature implementation across multiple layers (UI, business logic, data)
    - Find all related code that might be affected by feature changes
    - Understand how a feature is structured and organized
    
    Perfect for feature modification, debugging feature issues, or understanding
    complex feature implementations that span multiple modules.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        feature_keywords: List of keywords that identify the feature (e.g., ["login", "authenticate"])
        file_categories: Dictionary mapping category names to path patterns for file organization.
            Format: {"category_name": ["pattern1", "pattern2", ...]}
            
            Example:
            {
                "ui_frontend": ["view", "template", "component", "ui", "frontend"],
                "api_controllers": ["api", "controller", "endpoint", "route"],
                "business_logic": ["service", "business", "logic", "core"],
                "data_models": ["model", "entity", "schema", "db"],
                "utilities": ["util", "helper", "common"],
                "tests": ["test", "spec"],
                "config": ["config", "setting", "env"]
            }
            
            Files are categorized by checking if any pattern appears in the file path.
            Use descriptive category names that match your project structure.
        include_tests: Whether to include test files in the analysis
        include_config: Whether to include configuration files
        trace_depth: How deep to trace dependencies
    
    Returns:
        Complete feature implementation map with all involved components by layer
    """
    try:
        output_lines = ["=== Feature Implementation Analysis ===\n"]
        output_lines.append(f"Feature keywords: {', '.join(feature_keywords)}")
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Include tests: {include_tests}")
        output_lines.append(f"Include config: {include_config}")
        output_lines.append(f"Trace depth: {trace_depth}\n")
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
            import_graph = coderank_results["import_graph"]
        except:
            module_scores = {}
            import_graph = None
        
        # Find all files containing feature keywords
        all_matches = []
        
        for keyword in feature_keywords:
            # Search for the keyword in Python files
            rg_result = subprocess.run(
                ["rg", "-l", "-i", "--type", "py", keyword, repo_path],
                capture_output=True,
                text=True,
                cwd=working_directory
            )
            
            if rg_result.returncode == 0:
                for file_path in rg_result.stdout.strip().split('\n'):
                    if file_path:
                        all_matches.append(file_path)
        
        # Remove duplicates and categorize files
        unique_files = list(set(all_matches))
        
        # Categorize files using user-provided patterns
        categorized_files = {category: [] for category in file_categories.keys()}
        categorized_files['uncategorized'] = []
        
        for file_path in unique_files:
            rel_path = os.path.relpath(file_path, repo_path)
            lower_path = rel_path.lower()
            
            # Skip test files if not requested
            if not include_tests and any(test_pattern in lower_path for test_pattern in file_categories.get('tests', [])):
                continue
                
            # Skip config files if not requested  
            if not include_config and any(config_pattern in lower_path for config_pattern in file_categories.get('config', [])):
                continue
            
            # Categorize based on user-provided patterns
            categorized = False
            for category, patterns in file_categories.items():
                if any(pattern in lower_path for pattern in patterns):
                    categorized_files[category].append(file_path)
                    categorized = True
                    break
            
            if not categorized:
                categorized_files['uncategorized'].append(file_path)
        
        # For each category, show files with importance scores and key symbols
        for category, files in categorized_files.items():
            if not files:
                continue
                
            output_lines.append(f"\n=== {category.replace('_', ' ').title()} ({len(files)} files) ===")
            
            # Sort files by importance
            scored_files = []
            for file_path in files:
                module_fqn = path_to_module_fqn(file_path, repo_path)
                score = module_scores.get(module_fqn, 0) if module_fqn else 0
                scored_files.append((file_path, score))
            
            scored_files.sort(key=lambda x: x[1], reverse=True)
            
            for file_path, score in scored_files:
                rel_path = os.path.relpath(file_path, repo_path)
                importance = f" (importance: {score:.4f})" if score > 0 else ""
                output_lines.append(f"  📁 {rel_path}{importance}")
                
                # Get key symbols from this file related to the feature
                try:
                    symbols_output = get_repo_symbols(
                        repo=repo_path,
                        working_directory=working_directory,
                        file_must_contain=rel_path
                    )
                    
                    # Filter symbols that might be related to the feature
                    feature_symbols = []
                    for line in symbols_output:
                        if isinstance(line, str) and '|' in line and not line.startswith('-'):
                            for keyword in feature_keywords:
                                if keyword.lower() in line.lower():
                                    parts = re.split(r'\s{2,}', line)
                                    if len(parts) >= 2:
                                        symbol_name = parts[0]
                                        symbol_type = parts[1]
                                        feature_symbols.append(f"{symbol_type}: {symbol_name}")
                                    break
                    
                    if feature_symbols:
                        for symbol in feature_symbols[:3]:  # Show top 3 related symbols
                            output_lines.append(f"    • {symbol}")
                
                except Exception:
                    pass  # Skip symbol analysis if it fails
        
        # Analyze dependencies between feature files
        if import_graph:
            output_lines.append(f"\n=== Feature Dependencies ===")
            
            feature_modules = []
            for file_path in unique_files:
                module_fqn = path_to_module_fqn(file_path, repo_path)
                if module_fqn:
                    feature_modules.append(module_fqn)
            
            # Find dependencies between feature modules
            internal_deps = []
            external_deps = []
            
            for module in feature_modules:
                if module in import_graph:
                    # Check dependencies
                    for dependency in import_graph.successors(module):
                        if dependency in feature_modules:
                            internal_deps.append((module, dependency))
                        else:
                            # Only show high-importance external dependencies
                            dep_score = module_scores.get(dependency, 0)
                            if dep_score > 0.01:
                                external_deps.append((module, dependency, dep_score))
            
            if internal_deps:
                output_lines.append("Internal feature dependencies:")
                for source, target in internal_deps[:10]:
                    output_lines.append(f"  • {source} → {target}")
            
            if external_deps:
                output_lines.append("Important external dependencies:")
                external_deps.sort(key=lambda x: x[2], reverse=True)
                for source, target, score in external_deps[:5]:
                    output_lines.append(f"  • {source} → {target} (importance: {score:.4f})")
        
        # Find potential feature entry points
        output_lines.append(f"\n=== Potential Entry Points ===")
        entry_patterns = ["def.*" + "|".join(feature_keywords), "class.*" + "|".join(feature_keywords)]
        
        entry_points = []
        for pattern in entry_patterns:
            ep_result = subprocess.run(
                ["rg", "-n", "-i", "--type", "py", pattern, repo_path],
                capture_output=True,
                text=True,
                cwd=working_directory
            )
            
            if ep_result.returncode == 0:
                for line in ep_result.stdout.split('\n')[:5]:
                    if ':' in line:
                        entry_points.append(line)
        
        if entry_points:
            for entry in entry_points:
                output_lines.append(f"  • {entry}")
        else:
            output_lines.append("  • No obvious entry points found")
        
        # Summary
        output_lines.append(f"\n=== Feature Implementation Summary ===")
        total_files = len(unique_files)
        output_lines.append(f"Total files involved: {total_files}")
        
        # Count by category
        for category, files in categorized_files.items():
            if files:
                output_lines.append(f"  • {category.replace('_', ' ').title()}: {len(files)} files")
        
        # Recommendations based on categorized files
        output_lines.append(f"\n=== Modification Recommendations ===")
        for category, files in categorized_files.items():
            if files and category != 'uncategorized':
                output_lines.append(f"• {category.replace('_', ' ').title()}: {len(files)} files to review")
        
        if categorized_files.get('uncategorized'):
            output_lines.append(f"• Uncategorized: {len(categorized_files['uncategorized'])} files may need manual review")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in trace_feature_implementation: {str(e)}"


@mcp.tool()
def find_api_usage_examples(
    repo_path: str,
    working_directory: str,
    api_name: str,
    max_examples: int = 20,
    group_by_pattern: bool = True,
    include_test_examples: bool = True,
    show_context_lines: int = 5
) -> str:
    """
    Find real usage examples of APIs, functions, or classes in the codebase.
    
    Use this tool when you need to:
    - Learn how to properly use an existing API by seeing real examples
    - Understand the different ways an API is used across the codebase
    - Find patterns and best practices for API usage
    - See what parameters are commonly used and how
    
    Perfect for learning unfamiliar APIs, understanding usage patterns before
    making changes, or finding examples to follow when implementing similar functionality.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        api_name: Name of the API/function/class to find examples for
        max_examples: Maximum number of examples to return
        group_by_pattern: Whether to group similar usage patterns together
        include_test_examples: Whether to include examples from test files
        show_context_lines: Number of context lines around each usage
    
    Returns:
        Categorized real usage examples with context and patterns
    """
    try:
        output_lines = ["=== API Usage Examples ===\n"]
        output_lines.append(f"API: {api_name}")
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Max examples: {max_examples}")
        output_lines.append(f"Show context lines: {show_context_lines}\n")
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
        except:
            module_scores = {}
        
        # Use kit usages to find API usage sites
        try:
            usages = get_symbol_usages(
                repo=repo_path,
                symbol_name_or_substring=api_name,
                working_directory=working_directory
            )
        except Exception:
            # Fallback to ripgrep if kit usages fails
            rg_result = subprocess.run(
                ["rg", "-n", "-i", "--type", "py", api_name, repo_path],
                capture_output=True,
                text=True,
                cwd=working_directory
            )
            
            if rg_result.returncode != 0:
                return f"No usage examples found for: {api_name}"
            
            usages = rg_result.stdout.split('\n')
        
        # Parse and categorize usage examples
        examples = []
        
        for line in usages:
            if isinstance(line, str) and ':' in line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    file_path = parts[0]
                    line_number = int(parts[1]) if parts[1].isdigit() else 0
                    content = parts[2].strip()
                    
                    # Skip if test examples not wanted
                    if not include_test_examples and 'test' in file_path.lower():
                        continue
                    
                    # Get module importance
                    module_fqn = path_to_module_fqn(os.path.join(repo_path, file_path), repo_path)
                    score = module_scores.get(module_fqn, 0) if module_fqn else 0
                    
                    # Categorize usage pattern
                    usage_pattern = "function_call"
                    if f"{api_name}(" in content:
                        usage_pattern = "function_call"
                    elif f"import {api_name}" in content or f"from .* import.*{api_name}" in content:
                        usage_pattern = "import"
                    elif f"class.*{api_name}" in content:
                        usage_pattern = "inheritance"
                    elif f"{api_name} =" in content or f"= {api_name}" in content:
                        usage_pattern = "assignment"
                    elif f".{api_name}" in content:
                        usage_pattern = "method_call"
                    
                    examples.append({
                        'file': file_path,
                        'line': line_number,
                        'content': content,
                        'pattern': usage_pattern,
                        'score': score,
                        'is_test': 'test' in file_path.lower()
                    })
        
        if not examples:
            return f"No usage examples found for: {api_name}"
        
        # Sort by importance
        examples.sort(key=lambda x: x['score'], reverse=True)
        examples = examples[:max_examples]
        
        # Group by pattern if requested
        if group_by_pattern:
            patterns = {}
            for example in examples:
                pattern = example['pattern']
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(example)
            
            # Show examples grouped by pattern
            for pattern, pattern_examples in patterns.items():
                output_lines.append(f"\n=== {pattern.replace('_', ' ').title()} Examples ({len(pattern_examples)}) ===")
                
                for i, example in enumerate(pattern_examples[:10], 1):  # Max 10 per pattern
                    rel_path = os.path.relpath(example['file'], repo_path)
                    test_marker = " [TEST]" if example['is_test'] else ""
                    importance = f" (importance: {example['score']:.4f})" if example['score'] > 0 else ""
                    
                    output_lines.append(f"\n{i}. {rel_path}:{example['line']}{test_marker}{importance}")
                    output_lines.append(f"   {example['content']}")
                    
                    # Get context if requested
                    if show_context_lines > 0:
                        context = get_file_context(
                            os.path.join(repo_path, example['file']),
                            example['line'],
                            show_context_lines
                        )
                        if context:
                            output_lines.append("   Context:")
                            for ctx_line in context:
                                output_lines.append(f"   {ctx_line}")
        
        else:
            # Show examples in order of importance
            output_lines.append(f"\n=== Usage Examples (by importance) ===")
            
            for i, example in enumerate(examples, 1):
                rel_path = os.path.relpath(example['file'], repo_path)
                test_marker = " [TEST]" if example['is_test'] else ""
                importance = f" (importance: {example['score']:.4f})" if example['score'] > 0 else ""
                
                output_lines.append(f"\n{i}. {rel_path}:{example['line']}{test_marker}{importance}")
                output_lines.append(f"   Pattern: {example['pattern']}")
                output_lines.append(f"   {example['content']}")
                
                # Get context if requested
                if show_context_lines > 0:
                    context = get_file_context(
                        os.path.join(repo_path, example['file']),
                        example['line'],
                        show_context_lines
                    )
                    if context:
                        output_lines.append("   Context:")
                        for ctx_line in context:
                            output_lines.append(f"   {ctx_line}")
        
        # Usage pattern summary
        if group_by_pattern:
            output_lines.append(f"\n=== Usage Pattern Summary ===")
            pattern_counts = {}
            for example in examples:
                pattern = example['pattern']
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                output_lines.append(f"• {pattern.replace('_', ' ').title()}: {count} examples")
        
        # Recommendations
        output_lines.append(f"\n=== Usage Recommendations ===")
        
        # Find most common usage patterns
        function_calls = [ex for ex in examples if ex['pattern'] == 'function_call']
        if function_calls:
            output_lines.append(f"• Most common usage: function calls ({len(function_calls)} examples)")
            
            # Extract common parameter patterns
            param_patterns = set()
            for ex in function_calls[:5]:
                if '(' in ex['content'] and ')' in ex['content']:
                    params = ex['content'].split('(')[1].split(')')[0]
                    if params.strip():
                        param_patterns.add(params.strip())
            
            if param_patterns:
                output_lines.append("• Common parameter patterns:")
                for pattern in list(param_patterns)[:3]:
                    output_lines.append(f"  - {pattern}")
        
        # Check for consistent patterns
        high_importance_examples = [ex for ex in examples if ex['score'] > 0.01]
        if high_importance_examples:
            output_lines.append(f"• {len(high_importance_examples)} examples from high-importance modules")
            output_lines.append("• Follow patterns from these modules for best practices")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in find_api_usage_examples: {str(e)}"


def get_file_context(file_path: str, target_line: int, context_lines: int) -> List[str]:
    """Helper function to get context lines around a target line"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start = max(0, target_line - context_lines - 1)
        end = min(len(lines), target_line + context_lines)
        
        context = []
        for i in range(start, end):
            line_num = i + 1
            marker = ">>>" if line_num == target_line else "   "
            context.append(f"{marker} {line_num:3d}: {lines[i].rstrip()}")
        
        return context
    except Exception:
        return []


@mcp.tool()
def discover_side_effects(
    repo_path: str,
    working_directory: str,
    target_function: str,
    side_effect_patterns: Dict[str, List[str]],
    trace_depth: int = 3,
    include_historical_bugs: bool = True
) -> str:
    """
    Discover all potential side effects of calling a function or method.
    
    Use this tool when you need to understand:
    - What else might happen when you call a specific function
    - All the systems/resources that might be affected by a function call
    - Potential unintended consequences of code changes
    - What to test or monitor when modifying a function
    
    Critical for understanding the full impact of code changes, planning testing
    strategies, or debugging issues that might be caused by unexpected side effects.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        target_function: Name of the function to analyze for side effects
        side_effect_patterns: Dictionary mapping effect types to regex patterns.
            Format: {"category": ["regex1", "regex2", ...]}
            
            Example:
            {
                "file": [r"open\(", r"\.write\(", r"\.read\(", r"os\.remove", r"pathlib\."],
                "network": [r"requests\.", r"urllib\.", r"\.get\(", r"\.post\("],
                "database": [r"\.execute\(", r"\.query\(", r"\.commit\(", r"session\."],
                "global_state": [r"global ", r"os\.environ", r"setattr\("],
                "logging": [r"log\.", r"logger\.", r"print\("],
                "cache": [r"\.cache", r"redis\.", r"@lru_cache"]
            }
            
            Each regex pattern will be searched for in the function body to identify
            potential side effects. Use Python regex syntax.
        trace_depth: How deep to trace function calls for side effects
        include_historical_bugs: Whether to analyze git history for side effect bugs
    
    Returns:
        Comprehensive side effect analysis with risk assessment and mitigation suggestions
    """
    try:
        output_lines = ["=== Side Effect Discovery ===\n"]
        output_lines.append(f"Target function: {target_function}")
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Trace depth: {trace_depth}")
        output_lines.append(f"Side effect types: {', '.join(side_effect_patterns.keys())}\n")
        
        # First, find the target function definition
        func_def_result = subprocess.run(
            ["rg", "-n", "--type", "py", f"def {target_function}", repo_path],
            capture_output=True,
            text=True,
            cwd=working_directory
        )
        
        if func_def_result.returncode != 0:
            return f"Function '{target_function}' not found in repository"
        
        func_definitions = []
        for line in func_def_result.stdout.split('\n'):
            if ':' in line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    func_definitions.append({
                        'file': parts[0],
                        'line': int(parts[1]) if parts[1].isdigit() else 0,
                        'content': parts[2].strip()
                    })
        
        output_lines.append(f"Found {len(func_definitions)} definition(s) of '{target_function}':")
        for func_def in func_definitions:
            rel_path = os.path.relpath(func_def['file'], repo_path)
            output_lines.append(f"  • {rel_path}:{func_def['line']}")
        
        # Use the provided side effect patterns
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
            import_graph = coderank_results["import_graph"]
        except:
            module_scores = {}
            import_graph = None
        
        all_side_effects = {}
        
        # For each function definition, analyze for side effects
        for func_def in func_definitions:
            output_lines.append(f"\n=== Analyzing {os.path.relpath(func_def['file'], repo_path)} ===")
            
            # Get the function body (approximate by reading the file)
            try:
                with open(func_def['file'], 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Get function body (simplified - assumes standard indentation)
                start_line = func_def['line'] - 1
                func_body = []
                if start_line < len(lines):
                    # Find the function body by looking for consistent indentation
                    base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                    for i in range(start_line, min(len(lines), start_line + 50)):  # Max 50 lines
                        line = lines[i]
                        if line.strip() and not line.startswith('#'):
                            line_indent = len(line) - len(line.lstrip())
                            if i == start_line or line_indent > base_indent:
                                func_body.append(line.strip())
                            elif line_indent <= base_indent and i > start_line:
                                break
                
                func_content = '\n'.join(func_body)
                
            except Exception:
                func_content = ""
            
            # Search for side effects in the function body
            for effect_type, patterns in side_effect_patterns.items():
                effects_found = []
                
                for pattern in patterns:
                    if re.search(pattern, func_content, re.IGNORECASE):
                        effects_found.append(pattern)
                
                if effects_found:
                    if effect_type not in all_side_effects:
                        all_side_effects[effect_type] = []
                    all_side_effects[effect_type].extend(effects_found)
                    
                    output_lines.append(f"\n  {effect_type.upper()} side effects:")
                    for effect in effects_found:
                        output_lines.append(f"    • Pattern: {effect}")
            
            # Also search for function calls that might have side effects
            function_calls = re.findall(r'(\w+)\s*\(', func_content)
            if function_calls:
                output_lines.append(f"\n  Function calls (potential indirect side effects):")
                unique_calls = list(set(function_calls))[:10]  # Show unique calls, max 10
                for call in unique_calls:
                    output_lines.append(f"    • {call}()")
        
        # If we have import graph, trace dependencies for deeper side effect analysis
        if import_graph and trace_depth > 1:
            output_lines.append(f"\n=== Indirect Side Effects (via dependencies) ===")
            
            # Find modules that contain our target function
            target_modules = []
            for func_def in func_definitions:
                module_fqn = path_to_module_fqn(func_def['file'], repo_path)
                if module_fqn:
                    target_modules.append(module_fqn)
            
            # Trace dependencies up to trace_depth
            analyzed_modules = set(target_modules)
            current_level = target_modules
            
            for depth in range(1, trace_depth):
                next_level = set()
                for module in current_level:
                    if module in import_graph:
                        for dep in import_graph.successors(module):
                            if dep not in analyzed_modules:
                                next_level.add(dep)
                                analyzed_modules.add(dep)
                
                if next_level:
                    output_lines.append(f"\n  Level {depth} dependencies:")
                    for dep in sorted(next_level):
                        score = module_scores.get(dep, 0)
                        importance = f" (importance: {score:.4f})" if score > 0 else ""
                        output_lines.append(f"    • {dep}{importance}")
                        
                        # Quick side effect check on high-importance dependencies
                        if score > 0.01:
                            try:
                                dep_file = coderank_results["module_map"].get(dep)
                                if dep_file:
                                    # Quick pattern search
                                    quick_check = subprocess.run(
                                        ["rg", "-l", "--type", "py", "|".join([
                                            "open\\(", "requests\\.", "\.execute\\(", "log\\."
                                        ]), dep_file],
                                        capture_output=True,
                                        text=True,
                                        cwd=working_directory
                                    )
                                    if quick_check.returncode == 0:
                                        output_lines.append(f"      ⚠️  Potential side effects detected")
                            except:
                                pass
                
                current_level = next_level
                if not current_level:
                    break
        
        # Analyze historical bugs if requested
        if include_historical_bugs:
            output_lines.append(f"\n=== Historical Side Effect Issues ===")
            
            # Search for bug-related commits mentioning the function
            git_cmd = [
                "git", "-C", repo_path, "log",
                "--grep=bug",
                "--grep=fix",
                "--grep=issue",
                f"-S{target_function}",
                "--oneline",
                "--max-count=5"
            ]
            
            git_result = subprocess.run(git_cmd, capture_output=True, text=True)
            if git_result.returncode == 0 and git_result.stdout:
                commits = git_result.stdout.strip().split('\n')
                output_lines.append(f"Found {len(commits)} bug-related commits mentioning '{target_function}':")
                for commit in commits:
                    output_lines.append(f"  • {commit}")
            else:
                output_lines.append("No obvious bug-related commits found")
        
        # Risk assessment and recommendations
        output_lines.append(f"\n=== Side Effect Risk Assessment ===")
        
        total_side_effects = sum(len(effects) for effects in all_side_effects.values())
        output_lines.append(f"Total side effect patterns detected: {total_side_effects}")
        
        if total_side_effects == 0:
            output_lines.append("✅ No obvious side effects detected - relatively safe function")
        elif total_side_effects < 5:
            output_lines.append("⚠️  Low risk - Few side effects detected")
        elif total_side_effects < 10:
            output_lines.append("⚠️  Medium risk - Multiple side effects detected")
        else:
            output_lines.append("🚨 High risk - Many side effects detected")
        
        # Specific recommendations
        output_lines.append(f"\n=== Recommendations ===")
        
        if "file" in all_side_effects:
            output_lines.append("• File I/O detected - Test with different file conditions")
        if "network" in all_side_effects:
            output_lines.append("• Network calls detected - Test offline/timeout scenarios")
        if "database" in all_side_effects:
            output_lines.append("• Database operations detected - Test transaction rollback")
        if "global_state" in all_side_effects:
            output_lines.append("• Global state changes detected - Test state isolation")
        if "cache" in all_side_effects:
            output_lines.append("• Cache operations detected - Test cache invalidation")
        
        if total_side_effects > 0:
            output_lines.append("• Consider mocking external dependencies in tests")
            output_lines.append("• Monitor all affected systems when deploying changes")
            output_lines.append("• Document side effects for other developers")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in discover_side_effects: {str(e)}"


@mcp.tool()
def map_integration_points(
    repo_path: str,
    working_directory: str,
    integration_patterns: Dict[str, List[str]],
    include_error_handling: bool = True,
    show_configuration: bool = True,
    risk_assessment: bool = True
) -> str:
    """
    Map all external integration points and their characteristics.
    
    Use this tool when you need to understand:
    - What external services or systems the codebase integrates with
    - How those integrations are implemented and configured
    - Error handling patterns for external dependencies
    - Potential failure points and their impact
    
    Essential for understanding system dependencies, planning for service outages,
    or designing resilient integration patterns.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        integration_types: Types of integrations to look for (default: common types)
        custom_patterns: Custom regex patterns for integrations (overrides defaults)
        include_error_handling: Whether to analyze error handling for integrations
        show_configuration: Whether to find configuration related to integrations
        risk_assessment: Whether to assess risks of each integration
    
    Returns:
        Integration architecture map with dependency risks and patterns
    """
    try:
        # Remove hardcoded defaults - use provided patterns
        
        output_lines = ["=== Integration Points Analysis ===\n"]
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Integration types: {', '.join(integration_patterns.keys())}")
        output_lines.append(f"Include error handling: {include_error_handling}")
        output_lines.append(f"Show configuration: {show_configuration}\n")
        
        # Use the provided integration patterns
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
        except:
            module_scores = {}
        
        all_integrations = {}
        
        # Search for each integration type
        for integration_type, patterns in integration_patterns.items():
            integrations_found = []
            
            for pattern in patterns:
                rg_result = subprocess.run(
                    ["rg", "-n", "-i", "--type", "py", pattern, repo_path],
                    capture_output=True,
                    text=True,
                    cwd=working_directory
                )
                
                if rg_result.returncode == 0:
                    for line in rg_result.stdout.split('\n')[:10]:  # Top 10 per pattern
                        if ':' in line:
                            parts = line.split(':', 2)
                            if len(parts) >= 3:
                                file_path = parts[0]
                                line_number = parts[1]
                                content = parts[2].strip()
                                
                                # Get module importance
                                module_fqn = path_to_module_fqn(os.path.join(repo_path, file_path), repo_path)
                                score = module_scores.get(module_fqn, 0) if module_fqn else 0
                                
                                integrations_found.append({
                                    'file': file_path,
                                    'line': line_number,
                                    'content': content,
                                    'pattern': pattern,
                                    'score': score
                                })
            
            if integrations_found:
                all_integrations[integration_type] = integrations_found
        
        # Analyze and present results by integration type
        for integration_type, integrations in all_integrations.items():
            output_lines.append(f"\n=== {integration_type.upper()} Integrations ({len(integrations)} found) ===")
            
            # Sort by importance
            integrations.sort(key=lambda x: x['score'], reverse=True)
            
            # Group by file to show integration hotspots
            file_groups = {}
            for integration in integrations:
                file_path = integration['file']
                if file_path not in file_groups:
                    file_groups[file_path] = []
                file_groups[file_path].append(integration)
            
            # Show top integration files
            for file_path, file_integrations in sorted(
                file_groups.items(), 
                key=lambda x: sum(i['score'] for i in x[1]), 
                reverse=True
            )[:5]:  # Top 5 files per integration type
                rel_path = os.path.relpath(file_path, repo_path)
                total_score = sum(i['score'] for i in file_integrations)
                importance = f" (importance: {total_score:.4f})" if total_score > 0 else ""
                
                output_lines.append(f"\n  📁 {rel_path}{importance}")
                output_lines.append(f"     {len(file_integrations)} integration patterns found:")
                
                for integration in file_integrations[:3]:  # Top 3 patterns per file
                    output_lines.append(f"     • Line {integration['line']}: {integration['content'][:80]}...")
        
        # Look for error handling patterns if requested
        if include_error_handling:
            output_lines.append(f"\n=== Integration Error Handling ===")
            
            error_patterns = [
                r"try.*except.*request",
                r"try.*except.*connection",
                r"timeout",
                r"retry",
                r"fallback",
                r"circuit.*breaker"
            ]
            
            error_handling_found = []
            for pattern in error_patterns:
                eh_result = subprocess.run(
                    ["rg", "-n", "-i", "--type", "py", pattern, repo_path],
                    capture_output=True,
                    text=True,
                    cwd=working_directory
                )
                
                if eh_result.returncode == 0:
                    for line in eh_result.stdout.split('\n')[:5]:  # Top 5 per pattern
                        if line.strip():
                            error_handling_found.append(f"{pattern}: {line}")
            
            if error_handling_found:
                for error_handling in error_handling_found:
                    output_lines.append(f"  • {error_handling}")
            else:
                output_lines.append("  ⚠️  No obvious error handling patterns found")
        
        # Look for configuration if requested
        if show_configuration:
            output_lines.append(f"\n=== Integration Configuration ===")
            
            config_patterns = [
                r".*_URL",
                r".*_HOST",
                r".*_PORT",
                r".*_KEY",
                r".*_SECRET",
                r"DATABASE_URL",
                r"REDIS_URL",
                r"API_.*",
                r".*_ENDPOINT"
            ]
            
            config_found = []
            for pattern in config_patterns:
                cfg_result = subprocess.run(
                    ["rg", "-n", "-i", "--type", "py", pattern, repo_path],
                    capture_output=True,
                    text=True,
                    cwd=working_directory
                )
                
                if cfg_result.returncode == 0:
                    for line in cfg_result.stdout.split('\n')[:3]:  # Top 3 per pattern
                        if line.strip() and 'password' not in line.lower():  # Skip sensitive info
                            config_found.append(line)
            
            if config_found:
                for config in config_found[:10]:  # Top 10 config items
                    output_lines.append(f"  • {config}")
            else:
                output_lines.append("  • No obvious configuration patterns found")
        
        # Risk assessment if requested
        if risk_assessment:
            output_lines.append(f"\n=== Integration Risk Assessment ===")
            
            total_integrations = sum(len(integrations) for integrations in all_integrations.values())
            output_lines.append(f"Total integration points: {total_integrations}")
            
            # Risk factors
            risk_factors = []
            
            # Check for high integration counts per type
            for integration_type, integrations in all_integrations.items():
                count = len(integrations)
                if count > 10:
                    risk_factors.append(f"High number of {integration_type} integrations ({count}) - dependency risk")
                elif count > 5:
                    risk_factors.append(f"Multiple {integration_type} integrations ({count}) - coordination risk")
            
            if len(all_integrations) > 3:
                risk_factors.append("Multiple integration types - complexity risk")
            
            if not error_handling_found:
                risk_factors.append("Limited error handling patterns detected")
            
            if risk_factors:
                output_lines.append("\n  Risk factors identified:")
                for risk in risk_factors:
                    output_lines.append(f"  ⚠️  {risk}")
            else:
                output_lines.append("  ✅ No major risk factors identified")
        
        # Recommendations
        output_lines.append(f"\n=== Integration Recommendations ===")
        
        if all_integrations:
            output_lines.append("• Document all external dependencies and their SLAs")
            output_lines.append("• Implement circuit breakers for critical integrations")
            output_lines.append("• Add monitoring and alerting for integration failures")
            output_lines.append("• Consider fallback strategies for high-risk dependencies")
        
        # Generate recommendations based on integration types found
        for integration_type in all_integrations.keys():
            output_lines.append(f"• Review {integration_type} integration patterns for consistency")
            output_lines.append(f"• Consider error handling strategies for {integration_type} failures")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in map_integration_points: {str(e)}"


@mcp.tool()
def analyze_execution_paths(
    repo_path: str,
    working_directory: str,
    function_name: str,
    max_depth: int = 3,
    include_call_contexts: bool = True,
    highlight_complex_paths: bool = True
) -> str:
    """
    Analyze all possible execution paths through a function and what triggers each path.
    
    Use this tool when you need to understand:
    - All the different ways a complex function can execute
    - What conditions or parameters lead to different code paths
    - Potential edge cases or error conditions
    - Decision points and branching logic
    
    Perfect for understanding complex business logic, debugging function behavior,
    or planning test cases that cover all execution paths.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        function_name: Name of the function to analyze
        max_depth: How deep to analyze nested function calls
        include_call_contexts: Whether to show how the function is called
        highlight_complex_paths: Whether to identify complex/risky paths
    
    Returns:
        Execution path analysis with decision points and complexity assessment
    """
    try:
        output_lines = ["=== Execution Path Analysis ===\n"]
        output_lines.append(f"Function: {function_name}")
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Max depth: {max_depth}")
        output_lines.append(f"Include call contexts: {include_call_contexts}\n")
        
        # Find the function definition
        func_def_result = subprocess.run(
            ["rg", "-n", "--type", "py", f"def {function_name}", repo_path],
            capture_output=True,
            text=True,
            cwd=working_directory
        )
        
        if func_def_result.returncode != 0:
            return f"Function '{function_name}' not found in repository"
        
        func_definitions = []
        for line in func_def_result.stdout.split('\n'):
            if ':' in line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    func_definitions.append({
                        'file': parts[0],
                        'line': int(parts[1]) if parts[1].isdigit() else 0,
                        'content': parts[2].strip()
                    })
        
        output_lines.append(f"Found {len(func_definitions)} definition(s) of '{function_name}':")
        for func_def in func_definitions:
            rel_path = os.path.relpath(func_def['file'], repo_path)
            output_lines.append(f"  • {rel_path}:{func_def['line']}")
        
        # Analyze each function definition
        for func_def in func_definitions:
            output_lines.append(f"\n=== Analyzing {os.path.relpath(func_def['file'], repo_path)} ===")
            
            # Read the function body
            try:
                with open(func_def['file'], 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Extract function body
                start_line = func_def['line'] - 1
                func_body = []
                if start_line < len(lines):
                    base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                    for i in range(start_line, min(len(lines), start_line + 100)):  # Max 100 lines
                        line = lines[i]
                        if line.strip():
                            line_indent = len(line) - len(line.lstrip())
                            if i == start_line or line_indent > base_indent:
                                func_body.append((i + 1, line.rstrip()))
                            elif line_indent <= base_indent and i > start_line:
                                break
                
            except Exception:
                func_body = []
            
            if not func_body:
                output_lines.append("  Could not extract function body")
                continue
            
            # Analyze control flow structures
            decision_points = []
            complexity_score = 0
            nested_level = 0
            
            for line_num, line_content in func_body:
                stripped = line_content.strip()
                indent_level = len(line_content) - len(line_content.lstrip())
                
                # Track nesting level
                if any(keyword in stripped for keyword in ['if ', 'elif ', 'for ', 'while ', 'try:', 'except']):
                    nested_level = max(nested_level, indent_level)
                
                # Identify decision points
                if stripped.startswith('if ') or stripped.startswith('elif '):
                    condition = stripped[3:].rstrip(':') if stripped.startswith('if ') else stripped[5:].rstrip(':')
                    decision_points.append({
                        'line': line_num,
                        'type': 'conditional',
                        'condition': condition.strip(),
                        'complexity': 'simple' if len(condition) < 50 else 'complex'
                    })
                    complexity_score += 1
                
                elif stripped.startswith('for ') or stripped.startswith('while '):
                    loop_condition = stripped.split(':', 1)[0]
                    decision_points.append({
                        'line': line_num,
                        'type': 'loop',
                        'condition': loop_condition,
                        'complexity': 'simple'
                    })
                    complexity_score += 2  # Loops add more complexity
                
                elif stripped.startswith('try:'):
                    decision_points.append({
                        'line': line_num,
                        'type': 'exception_handling',
                        'condition': 'try block',
                        'complexity': 'simple'
                    })
                    complexity_score += 1
                
                elif stripped.startswith('except'):
                    exception_type = stripped.split(':', 1)[0]
                    decision_points.append({
                        'line': line_num,
                        'type': 'exception_handling',
                        'condition': exception_type,
                        'complexity': 'simple'
                    })
                
                elif 'return ' in stripped:
                    return_condition = "early return" if line_num < func_body[-1][0] else "final return"
                    decision_points.append({
                        'line': line_num,
                        'type': 'return',
                        'condition': return_condition,
                        'complexity': 'simple'
                    })
            
            # Display decision points
            output_lines.append(f"\n  Decision Points ({len(decision_points)} found):")
            if decision_points:
                for dp in decision_points:
                    complexity_marker = "🔥" if dp['complexity'] == 'complex' else "•"
                    output_lines.append(f"    {complexity_marker} Line {dp['line']}: {dp['type'].title()} - {dp['condition'][:60]}")
            else:
                output_lines.append("    • No decision points found - linear execution")
            
            # Analyze function calls that might add complexity
            function_calls = []
            for line_num, line_content in func_body:
                # Find function calls
                import re
                calls = re.findall(r'(\w+)\s*\(', line_content)
                for call in calls:
                    if call not in ['print', 'len', 'str', 'int', 'float', 'bool']:  # Skip built-ins
                        function_calls.append({
                            'line': line_num,
                            'function': call
                        })
            
            if function_calls:
                output_lines.append(f"\n  Function Calls ({len(function_calls)} found):")
                unique_calls = {}
                for call in function_calls:
                    func = call['function']
                    if func not in unique_calls:
                        unique_calls[func] = []
                    unique_calls[func].append(call['line'])
                
                for func, lines in sorted(unique_calls.items())[:10]:  # Top 10
                    line_list = ', '.join(map(str, lines[:3]))
                    more = f" +{len(lines)-3} more" if len(lines) > 3 else ""
                    output_lines.append(f"    • {func}() at lines: {line_list}{more}")
            
            # Complexity assessment
            output_lines.append(f"\n  Complexity Assessment:")
            output_lines.append(f"    • Cyclomatic complexity: {complexity_score + 1}")
            output_lines.append(f"    • Max nesting level: {nested_level // 4}")  # Approximate based on indentation
            output_lines.append(f"    • Decision points: {len(decision_points)}")
            output_lines.append(f"    • Function calls: {len(function_calls)}")
            
            if highlight_complex_paths:
                complex_conditions = [dp for dp in decision_points if dp['complexity'] == 'complex']
                if complex_conditions:
                    output_lines.append(f"\n  🔥 Complex conditions requiring attention:")
                    for condition in complex_conditions:
                        output_lines.append(f"    • Line {condition['line']}: {condition['condition']}")
        
        # Show function usage contexts if requested
        if include_call_contexts:
            output_lines.append(f"\n=== Function Usage Contexts ===")
            
            # Find how this function is called
            try:
                usages = get_symbol_usages(
                    repo=repo_path,
                    symbol_name_or_substring=function_name,
                    working_directory=working_directory
                )
                
                usage_contexts = []
                for line in usages:
                    if isinstance(line, str) and ':' in line and f"{function_name}(" in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            usage_contexts.append({
                                'file': parts[0],
                                'line': parts[1],
                                'context': parts[2].strip()
                            })
                
                if usage_contexts:
                    output_lines.append(f"Function called from {len(usage_contexts)} locations:")
                    for usage in usage_contexts[:5]:  # Top 5 usage contexts
                        rel_path = os.path.relpath(usage['file'], repo_path)
                        output_lines.append(f"  • {rel_path}:{usage['line']}")
                        output_lines.append(f"    {usage['context'][:80]}...")
                else:
                    output_lines.append("No usage contexts found")
                    
            except Exception:
                output_lines.append("Could not analyze usage contexts")
        
        # Recommendations
        output_lines.append(f"\n=== Recommendations ===")
        
        if complexity_score > 10:
            output_lines.append("• Consider breaking this function into smaller functions")
            output_lines.append("• Add comprehensive unit tests covering all paths")
            output_lines.append("• Document complex decision logic")
        elif complexity_score > 5:
            output_lines.append("• Add tests for the main execution paths")
            output_lines.append("• Consider adding more documentation")
        else:
            output_lines.append("• Function complexity is manageable")
        
        if decision_points:
            output_lines.append(f"• Test all {len(decision_points)} decision points")
            output_lines.append("• Verify error handling paths work correctly")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in analyze_execution_paths: {str(e)}"


@mcp.tool()
def analyze_config_impact(
    repo_path: str,
    working_directory: str,
    config_key: str,
    trace_dependent_logic: bool = True,
    include_default_handling: bool = True,
    show_historical_changes: bool = True
) -> str:
    """
    Analyze what code is affected by specific configuration values.
    
    Use this tool when you need to understand:
    - What code will behave differently if you change a config setting
    - How configuration values flow through the system
    - What the default behavior is when config is missing
    - How configuration changes have been handled historically
    
    Critical for understanding the impact of configuration modifications,
    planning configuration changes, or debugging configuration-related issues.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        config_key: Configuration key to analyze (e.g., "DEBUG", "DATABASE_URL")
        trace_dependent_logic: Whether to trace through conditional logic
        include_default_handling: Whether to analyze default value handling
        show_historical_changes: Whether to show config-related git history
    
    Returns:
        Configuration impact analysis with affected code paths and recommendations
    """
    try:
        output_lines = ["=== Configuration Impact Analysis ===\n"]
        output_lines.append(f"Configuration key: {config_key}")
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Trace dependent logic: {trace_dependent_logic}")
        output_lines.append(f"Include defaults: {include_default_handling}\n")
        
        # Find all references to the configuration key
        config_usages = []
        
        # Search for direct references
        rg_result = subprocess.run(
            ["rg", "-n", "-i", "--type", "py", config_key, repo_path],
            capture_output=True,
            text=True,
            cwd=working_directory
        )
        
        if rg_result.returncode != 0:
            return f"No references found for configuration key: {config_key}"
        
        # Parse configuration usages
        for line in rg_result.stdout.split('\n'):
            if ':' in line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    file_path = parts[0]
                    line_number = parts[1]
                    content = parts[2].strip()
                    
                    # Categorize the usage
                    usage_type = "reference"
                    if any(pattern in content.lower() for pattern in ['=', 'get(', 'getenv', 'config']):
                        if f"{config_key} =" in content or f"= {config_key}" in content:
                            usage_type = "assignment"
                        elif 'get(' in content or 'getenv' in content:
                            usage_type = "retrieval"
                        elif 'config' in content.lower():
                            usage_type = "configuration"
                    
                    if any(pattern in content.lower() for pattern in ['if ', 'elif ', 'when ', '?']):
                        usage_type = "conditional"
                    
                    config_usages.append({
                        'file': file_path,
                        'line': line_number,
                        'content': content,
                        'type': usage_type
                    })
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
        except:
            module_scores = {}
        
        # Group usages by type and analyze
        usage_types = {}
        for usage in config_usages:
            usage_type = usage['type']
            if usage_type not in usage_types:
                usage_types[usage_type] = []
            usage_types[usage_type].append(usage)
        
        # Analyze each usage type
        for usage_type, usages in usage_types.items():
            output_lines.append(f"\n=== {usage_type.title()} Usage ({len(usages)} found) ===")
            
            # Sort by module importance
            scored_usages = []
            for usage in usages:
                module_fqn = path_to_module_fqn(os.path.join(repo_path, usage['file']), repo_path)
                score = module_scores.get(module_fqn, 0) if module_fqn else 0
                scored_usages.append((usage, score))
            
            scored_usages.sort(key=lambda x: x[1], reverse=True)
            
            for usage, score in scored_usages[:10]:  # Top 10 per type
                rel_path = os.path.relpath(usage['file'], repo_path)
                importance = f" (importance: {score:.4f})" if score > 0 else ""
                output_lines.append(f"  • {rel_path}:{usage['line']}{importance}")
                output_lines.append(f"    {usage['content']}")
        
        # Trace dependent logic if requested
        if trace_dependent_logic:
            output_lines.append(f"\n=== Dependent Logic Analysis ===")
            
            # Look for conditional statements that depend on this config
            conditional_usages = [u for u in config_usages if u['type'] == 'conditional']
            
            if conditional_usages:
                output_lines.append(f"Found {len(conditional_usages)} conditional logic blocks:")
                
                for usage in conditional_usages[:5]:  # Top 5 conditionals
                    rel_path = os.path.relpath(usage['file'], repo_path)
                    output_lines.append(f"\n  📍 {rel_path}:{usage['line']}")
                    output_lines.append(f"     {usage['content']}")
                    
                    # Try to get context around the conditional
                    try:
                        context = get_file_context(
                            os.path.join(repo_path, usage['file']),
                            int(usage['line']),
                            3
                        )
                        if context:
                            output_lines.append("     Context:")
                            for ctx_line in context:
                                output_lines.append(f"     {ctx_line}")
                    except:
                        pass
            else:
                output_lines.append("No conditional logic directly dependent on this config found")
            
            # Look for functions that might be affected
            affected_functions = set()
            for usage in config_usages:
                try:
                    with open(os.path.join(repo_path, usage['file']), 'r') as f:
                        lines = f.readlines()
                    
                    usage_line = int(usage['line']) - 1
                    # Look backwards to find the containing function
                    for i in range(usage_line, max(0, usage_line - 20), -1):
                        if lines[i].strip().startswith('def '):
                            func_name = lines[i].strip().split('(')[0].replace('def ', '')
                            affected_functions.add(func_name)
                            break
                except:
                    pass
            
            if affected_functions:
                output_lines.append(f"\n  Functions potentially affected by config changes:")
                for func in sorted(affected_functions)[:10]:
                    output_lines.append(f"    • {func}()")
        
        # Analyze default handling if requested
        if include_default_handling:
            output_lines.append(f"\n=== Default Value Analysis ===")
            
            # Look for default value patterns
            default_patterns = [
                r"\.get\(['\"]" + re.escape(config_key) + r"['\"],\s*['\"]?([^'\")\]]+)",
                r"getenv\(['\"]" + re.escape(config_key) + r"['\"],\s*['\"]?([^'\")\]]+)",
                r"os\.environ\.get\(['\"]" + re.escape(config_key) + r"['\"],\s*['\"]?([^'\")\]]+)",
                r"or\s+['\"]([^'\"]+)['\"]",  # Common pattern: config_value or "default"
            ]
            
            defaults_found = []
            for pattern in default_patterns:
                default_result = subprocess.run(
                    ["rg", "-n", "--type", "py", pattern, repo_path],
                    capture_output=True,
                    text=True,
                    cwd=working_directory
                )
                
                if default_result.returncode == 0:
                    for line in default_result.stdout.split('\n')[:3]:  # Top 3 per pattern
                        if line.strip():
                            defaults_found.append(line)
            
            if defaults_found:
                output_lines.append("Default value patterns found:")
                for default in defaults_found:
                    output_lines.append(f"  • {default}")
            else:
                output_lines.append("⚠️  No obvious default value handling found")
                output_lines.append("  Consider what happens when this config is missing")
        
        # Show historical changes if requested
        if show_historical_changes:
            output_lines.append(f"\n=== Historical Configuration Changes ===")
            
            # Search git history for changes to this config
            git_cmd = [
                "git", "-C", repo_path, "log",
                "--grep=" + config_key,
                f"-S{config_key}",
                "--oneline",
                "--max-count=10"
            ]
            
            git_result = subprocess.run(git_cmd, capture_output=True, text=True)
            if git_result.returncode == 0 and git_result.stdout:
                commits = git_result.stdout.strip().split('\n')
                output_lines.append(f"Found {len(commits)} commits related to this config:")
                for commit in commits:
                    output_lines.append(f"  • {commit}")
            else:
                output_lines.append("No historical changes found for this configuration")
        
        # Impact assessment
        output_lines.append(f"\n=== Impact Assessment ===")
        
        total_usages = len(config_usages)
        unique_files = len(set(usage['file'] for usage in config_usages))
        conditional_count = len([u for u in config_usages if u['type'] == 'conditional'])
        
        output_lines.append(f"Configuration impact scope:")
        output_lines.append(f"  • Total references: {total_usages}")
        output_lines.append(f"  • Files affected: {unique_files}")
        output_lines.append(f"  • Conditional logic blocks: {conditional_count}")
        
        if not defaults_found:
            output_lines.append("• Add default value handling to prevent failures")
        
        if conditional_count > 0:
            output_lines.append(f"• Test both true/false cases for {conditional_count} conditional blocks")
        
        output_lines.append("• Update documentation if config behavior changes")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in analyze_config_impact: {str(e)}"


@mcp.tool()
def identify_performance_bottlenecks(
    repo_path: str,
    working_directory: str,
    bottleneck_patterns: Dict[str, List[str]],
    include_usage_frequency: bool = True,
    days_back: int = 90
) -> str:
    """
    Identify potential performance bottlenecks by analyzing code patterns and complexity.
    
    Use this tool when you need to:
    - Find likely performance issues before they become problems
    - Prioritize optimization efforts on high-impact code
    - Understand which parts of the codebase might be slow
    - Plan performance improvements based on actual usage patterns
    
    Perfect for performance optimization planning, code reviews focused on performance,
    or debugging existing performance issues.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        focus_areas: Specific performance areas to focus on (default: common bottlenecks)
        custom_patterns: Custom regex patterns for bottlenecks (overrides defaults)
        include_usage_frequency: Whether to weight results by code usage frequency
        days_back: Days of git history to analyze for frequently changed performance code
    
    Returns:
        Performance bottleneck analysis with optimization recommendations
    """
    try:
        output_lines = ["=== Performance Bottleneck Analysis ===\n"]
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Focus areas: {', '.join(bottleneck_patterns.keys())}")
        output_lines.append(f"Include usage frequency: {include_usage_frequency}")
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
        except:
            module_scores = {}
        
        all_bottlenecks = {}
        
        # Search for each bottleneck pattern
        for area, patterns in bottleneck_patterns.items():
            bottlenecks_found = []
            
            for pattern in patterns:
                rg_result = subprocess.run(
                    ["rg", "-n", "--type", "py", pattern, repo_path],
                    capture_output=True,
                    text=True,
                    cwd=working_directory
                )
                
                if rg_result.returncode == 0:
                    for line in rg_result.stdout.split('\n')[:10]:  # Top 10 per pattern
                        if ':' in line:
                            parts = line.split(':', 2)
                            if len(parts) >= 3:
                                file_path = parts[0]
                                line_number = parts[1]
                                content = parts[2].strip()
                                
                                # Get module importance
                                module_fqn = path_to_module_fqn(os.path.join(repo_path, file_path), repo_path)
                                score = module_scores.get(module_fqn, 0) if module_fqn else 0
                                
                                bottlenecks_found.append({
                                    'file': file_path,
                                    'line': line_number,
                                    'content': content,
                                    'pattern': pattern,
                                    'score': score
                                })
            
            if bottlenecks_found:
                all_bottlenecks[area] = bottlenecks_found
        
        # Analyze and present results
        for area, bottlenecks in all_bottlenecks.items():
            output_lines.append(f"\n=== {area.upper()} Performance Issues ({len(bottlenecks)} found) ===")
            
            # Sort by importance and deduplicate by file
            bottlenecks.sort(key=lambda x: x['score'], reverse=True)
            
            # Group by file to show hotspot files
            file_groups = {}
            for bottleneck in bottlenecks:
                file_path = bottleneck['file']
                if file_path not in file_groups:
                    file_groups[file_path] = []
                file_groups[file_path].append(bottleneck)
            
            # Show top problematic files
            for file_path, file_bottlenecks in sorted(
                file_groups.items(),
                key=lambda x: sum(b['score'] for b in x[1]),
                reverse=True
            )[:5]:  # Top 5 files per area
                rel_path = os.path.relpath(file_path, repo_path)
                total_score = sum(b['score'] for b in file_bottlenecks)
                importance = f" (importance: {total_score:.4f})" if total_score > 0 else ""
                
                output_lines.append(f"\n  🐌 {rel_path}{importance}")
                output_lines.append(f"     {len(file_bottlenecks)} potential issues:")
                
                for bottleneck in file_bottlenecks[:3]:  # Top 3 issues per file
                    output_lines.append(f"     • Line {bottleneck['line']}: {bottleneck['content'][:80]}...")
        
        # Analyze frequently changed performance-sensitive code
        if include_usage_frequency:
            output_lines.append(f"\n=== Frequently Modified Performance Code ===")
            
            since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Get files that have been changed frequently and have performance issues
            performance_files = set()
            for bottlenecks in all_bottlenecks.values():
                for bottleneck in bottlenecks:
                    performance_files.add(bottleneck['file'])
            
            frequently_changed = []
            for file_path in performance_files:
                rel_path = os.path.relpath(file_path, repo_path)
                git_cmd = [
                    "git", "-C", repo_path, "log",
                    f"--since={since_date}",
                    "--oneline",
                    "--", rel_path
                ]
                
                git_result = subprocess.run(git_cmd, capture_output=True, text=True)
                if git_result.returncode == 0:
                    commit_count = len([line for line in git_result.stdout.split('\n') if line.strip()])
                    if commit_count > 3:  # Files changed more than 3 times
                        module_fqn = path_to_module_fqn(file_path, repo_path)
                        score = module_scores.get(module_fqn, 0) if module_fqn else 0
                        frequently_changed.append({
                            'file': file_path,
                            'commits': commit_count,
                            'score': score
                        })
            
            if frequently_changed:
                frequently_changed.sort(key=lambda x: (x['score'], x['commits']), reverse=True)
                output_lines.append("Performance-sensitive files with frequent changes:")
                for item in frequently_changed[:5]:
                    rel_path = os.path.relpath(item['file'], repo_path)
                    output_lines.append(f"  • {rel_path}: {item['commits']} commits (importance: {item['score']:.4f})")
            else:
                output_lines.append("No frequently changed performance-sensitive files found")
        
        # Performance risk assessment
        output_lines.append(f"\n=== Performance Risk Assessment ===")
        
        total_issues = sum(len(bottlenecks) for bottlenecks in all_bottlenecks.values())
        high_impact_issues = sum(
            1 for bottlenecks in all_bottlenecks.values()
            for bottleneck in bottlenecks
            if bottleneck['score'] > 0.01
        )
        
        output_lines.append(f"Performance analysis summary:")
        output_lines.append(f"  • Total potential issues: {total_issues}")
        output_lines.append(f"  • High-impact issues: {high_impact_issues}")
        output_lines.append(f"  • Areas of concern: {len(all_bottlenecks)}")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in identify_performance_bottlenecks: {str(e)}"


@mcp.tool()
def analyze_testing_strategy(
    repo_path: str,
    working_directory: str,
    test_file_patterns: List[str],
    framework_patterns: Optional[Dict[str, List[str]]] = None,
    test_types: List[str] = None,
    show_coverage_gaps: bool = True,
    include_testing_patterns: bool = True,
    focus_on_important_modules: bool = True
) -> str:
    """
    Analyze testing strategies and identify gaps for better test coverage.
    
    Use this tool when you need to understand:
    - How different parts of the code are tested and what patterns are used
    - Where test coverage might be missing or insufficient
    - What testing frameworks and patterns the codebase follows
    - How to write tests that fit the existing testing strategy
    
    Perfect for understanding how to test new code, improving test coverage,
    or learning the testing patterns used in an unfamiliar codebase.
    
    Args:
        repo_path: Repository path (absolute)
        working_directory: Working directory (absolute path)
        test_types: Types of tests to analyze (default: common test types)
        custom_file_patterns: Custom regex patterns for test files (overrides defaults)
        custom_framework_patterns: Custom regex patterns for frameworks (overrides defaults)
        show_coverage_gaps: Whether to identify modules that might lack tests
        include_testing_patterns: Whether to analyze testing patterns and frameworks
        focus_on_important_modules: Whether to prioritize important modules in analysis
    
    Returns:
        Testing strategy analysis with patterns, gaps, and recommendations
    """
    try:
        if test_types is None:
            test_types = ["unit", "integration", "e2e", "functional"]
        
        output_lines = ["=== Testing Strategy Analysis ===\n"]
        output_lines.append(f"Repository: {repo_path}")
        output_lines.append(f"Test types: {', '.join(test_types)}")
        output_lines.append(f"Show coverage gaps: {show_coverage_gaps}")
        output_lines.append(f"Include patterns: {include_testing_patterns}\n")
        
        # Get CodeRank data for prioritization
        try:
            coderank_results = calculate_coderank(repo_path=repo_path)
            module_scores = coderank_results["module_ranks"]
        except:
            module_scores = {}
        
        # Use the provided test file patterns
        
        test_files = []
        for pattern in test_file_patterns:
            rg_result = subprocess.run(
                ["rg", "-l", "--type", "py", pattern, repo_path],
                capture_output=True,
                text=True,
                cwd=working_directory
            )
            
            if rg_result.returncode == 0:
                for file_path in rg_result.stdout.strip().split('\n'):
                    if file_path and file_path not in test_files:
                        test_files.append(file_path)
        
        output_lines.append(f"Found {len(test_files)} test files")
        
        # Analyze testing frameworks and patterns
        if include_testing_patterns:
            output_lines.append(f"\n=== Testing Frameworks & Patterns ===")
            
            if not framework_patterns:
                output_lines.append("No framework patterns provided for analysis")
                frameworks_found = {}
            else:
                frameworks_found = {}
                for framework, patterns in framework_patterns.items():
                    framework_usage = []
                    for pattern in patterns:
                        rg_result = subprocess.run(
                            ["rg", "-c", "--type", "py", pattern, repo_path],
                            capture_output=True,
                            text=True,
                            cwd=working_directory
                        )
                        
                        if rg_result.returncode == 0:
                            count = sum(int(line.split(':')[1]) for line in rg_result.stdout.split('\n') if ':' in line)
                            if count > 0:
                                framework_usage.append(count)
                    
                    if framework_usage:
                        frameworks_found[framework] = sum(framework_usage)
            
            if frameworks_found:
                output_lines.append("Testing frameworks in use:")
                for framework, count in sorted(frameworks_found.items(), key=lambda x: x[1], reverse=True):
                    output_lines.append(f"  • {framework}: {count} usages")
            else:
                output_lines.append("No obvious testing frameworks detected")
        
        # Analyze test coverage patterns
        test_function_count = 0
        test_methods = []
        
        for test_file in test_files[:20]:  # Analyze top 20 test files
            try:
                # Get test functions/methods from the file
                symbols_output = get_repo_symbols(
                    repo=repo_path,
                    working_directory=working_directory,
                    file_must_contain=os.path.relpath(test_file, repo_path),
                    keep_types=["function", "method"]
                )
                
                for line in symbols_output:
                    if isinstance(line, str) and '|' in line and not line.startswith('-'):
                        parts = re.split(r'\s{2,}', line)
                        if len(parts) >= 2:
                            symbol_name = parts[0]
                            symbol_type = parts[1]
                            if symbol_name.startswith('test_') or 'test' in symbol_name.lower():
                                test_methods.append({
                                    'name': symbol_name,
                                    'type': symbol_type,
                                    'file': test_file
                                })
                                test_function_count += 1
                                
            except Exception:
                pass
        
        output_lines.append(f"\n=== Test Coverage Analysis ===")
        output_lines.append(f"Total test functions/methods found: {test_function_count}")
        
        # Categorize tests by apparent type
        test_categories = {
            "unit": [],
            "integration": [],
            "e2e": [],
            "other": []
        }
        
        for test_method in test_methods:
            name = test_method['name'].lower()
            if any(keyword in name for keyword in ['unit', 'mock', 'stub']):
                test_categories["unit"].append(test_method)
            elif any(keyword in name for keyword in ['integration', 'api', 'db', 'database']):
                test_categories["integration"].append(test_method)
            elif any(keyword in name for keyword in ['e2e', 'end_to_end', 'full', 'scenario']):
                test_categories["e2e"].append(test_method)
            else:
                test_categories["other"].append(test_method)
        
        output_lines.append("\nTest distribution by apparent type:")
        for category, tests in test_categories.items():
            if tests:
                output_lines.append(f"  • {category.title()}: {len(tests)} tests")
        
        # Show coverage gaps if requested
        if show_coverage_gaps:
            output_lines.append(f"\n=== Coverage Gap Analysis ===")
            
            # Find production modules (non-test Python files)
            production_files = []
            all_py_result = subprocess.run(
                ["rg", "-l", "--type", "py", ".", repo_path],
                capture_output=True,
                text=True,
                cwd=working_directory
            )
            
            if all_py_result.returncode == 0:
                for file_path in all_py_result.stdout.strip().split('\n'):
                    if file_path and not any(test_pattern in file_path.lower() for test_pattern in ['test', 'spec']):
                        production_files.append(file_path)
            
            # Identify modules that might lack tests
            untested_modules = []
            for prod_file in production_files:
                # Look for corresponding test file
                rel_path = os.path.relpath(prod_file, repo_path)
                module_name = os.path.splitext(os.path.basename(rel_path))[0]
                
                # Check various test naming conventions
                has_test = False
                test_patterns_to_check = [
                    f"test_{module_name}",
                    f"{module_name}_test",
                    f"Test{module_name.title()}",
                    module_name
                ]
                
                for test_file in test_files:
                    test_content = os.path.basename(test_file).lower()
                    if any(pattern.lower() in test_content for pattern in test_patterns_to_check):
                        has_test = True
                        break
                
                if not has_test and focus_on_important_modules:
                    module_fqn = path_to_module_fqn(prod_file, repo_path)
                    score = module_scores.get(module_fqn, 0) if module_fqn else 0
                    if score > 0.01:  # Only include important modules
                        untested_modules.append({
                            'file': rel_path,
                            'module': module_fqn or rel_path,
                            'score': score
                        })
                elif not has_test and not focus_on_important_modules:
                    untested_modules.append({
                        'file': rel_path,
                        'module': rel_path,
                        'score': 0
                    })
            
            if untested_modules:
                untested_modules.sort(key=lambda x: x['score'], reverse=True)
                output_lines.append(f"Modules that may lack tests ({len(untested_modules)} found):")
                for module in untested_modules[:10]:  # Top 10
                    importance = f" (importance: {module['score']:.4f})" if module['score'] > 0 else ""
                    output_lines.append(f"  ⚠️  {module['file']}{importance}")
            else:
                output_lines.append("✅ No obvious testing gaps found")
        
        # Analyze test quality indicators
        output_lines.append(f"\n=== Test Quality Indicators ===")
        
        quality_patterns = {
            "assertions": [r"assert ", r"assertEqual", r"assertTrue", r"assertFalse"],
            "mocking": [r"@mock\.", r"Mock\(", r"patch\("],
            "setup_teardown": [r"setUp", r"tearDown", r"@pytest\.fixture"],
            "error_testing": [r"assertRaises", r"pytest\.raises", r"with.*raises"]
        }
        
        quality_scores = {}
        for indicator, patterns in quality_patterns.items():
            total_count = 0
            for pattern in patterns:
                rg_result = subprocess.run(
                    ["rg", "-c", "--type", "py", pattern, repo_path],
                    capture_output=True,
                    text=True,
                    cwd=working_directory
                )
                
                if rg_result.returncode == 0:
                    count = sum(int(line.split(':')[1]) for line in rg_result.stdout.split('\n') if ':' in line)
                    total_count += count
            
            quality_scores[indicator] = total_count
        
        output_lines.append("Test quality indicators:")
        for indicator, count in quality_scores.items():
            status = "✅" if count > 10 else "⚠️" if count > 0 else "❌"
            output_lines.append(f"  {status} {indicator.replace('_', ' ').title()}: {count}")
        
        # Testing strategy recommendations
        output_lines.append(f"\n=== Testing Strategy Recommendations ===")
        
        # Based on framework usage
        if frameworks_found:
            primary_framework = max(frameworks_found.items(), key=lambda x: x[1])[0]
            output_lines.append(f"• Primary testing framework: {primary_framework}")
            output_lines.append(f"• Follow existing {primary_framework} patterns for consistency")
        else:
            output_lines.append("• Consider adopting a standard testing framework (pytest recommended)")
        
        # Based on test distribution
        total_tests = sum(len(tests) for tests in test_categories.values())
        if total_tests > 0:
            unit_ratio = len(test_categories["unit"]) / total_tests
            if unit_ratio < 0.7:
                output_lines.append("• Consider adding more unit tests (recommended: 70% of tests)")
            
            integration_ratio = len(test_categories["integration"]) / total_tests
            if integration_ratio > 0.3:
                output_lines.append("• Good integration test coverage - maintain this balance")
        
        # Based on coverage gaps
        if untested_modules and len(untested_modules) > 5:
            output_lines.append(f"• Priority: Add tests for {len(untested_modules)} untested modules")
            output_lines.append("• Start with the highest importance modules")
        
        # Based on quality indicators
        if quality_scores.get("assertions", 0) < total_tests:
            output_lines.append("• Add more assertions to verify test outcomes")
        
        if quality_scores.get("error_testing", 0) < 5:
            output_lines.append("• Add tests for error conditions and edge cases")
        
        if quality_scores.get("mocking", 0) < 5:
            output_lines.append("• Consider using mocks to isolate units under test")
        
        # Test coverage ratio
        if production_files and test_files:
            test_ratio = len(test_files) / len(production_files)
            output_lines.append(f"\n=== Test Coverage Ratio ===")
            output_lines.append(f"Test files to production files ratio: {test_ratio:.2f}")
            
            if test_ratio < 0.3:
                output_lines.append("⚠️  Low test coverage ratio - consider adding more tests")
            elif test_ratio > 0.8:
                output_lines.append("✅ Good test coverage ratio")
            else:
                output_lines.append("📊 Moderate test coverage ratio")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in analyze_testing_strategy: {str(e)}"


# helper function not exposed as a tool
def path_to_module_fqn(file_path: str, repo_path: str) -> Optional[str]:
    """Convert file path to module FQN - reuse from coderank.py"""
    from coderank import path_to_module_fqn as _path_to_module_fqn
    return _path_to_module_fqn(file_path, repo_path)
