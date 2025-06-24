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
                risk_level = "VERY HIGH" if target_score > 0.01 or len(downstream_deps) > 5 else "HIGH"
                
            elif change_type == "split":
                output_lines.append(f"  • Splitting will require updating imports in {len(downstream_deps)} modules")
                output_lines.append(f"  • {symbol_count} symbols need to be redistributed")
                output_lines.append(f"  • Consider grouping by: functionality, dependencies, or usage patterns")
                risk_level = "MEDIUM" if len(downstream_deps) < 10 else "HIGH"
                
            elif change_type == "merge":
                output_lines.append(f"  • Merging will consolidate {symbol_count} symbols")
                output_lines.append(f"  • May increase module complexity")
                connections = import_graph.degree(target_module)
                output_lines.append(f"  • Current module connections: {connections}")
                risk_level = "LOW" if symbol_count < 20 else "MEDIUM"
                
            else:  # modify
                output_lines.append(f"  • Modifications will affect {len(downstream_deps)} dependent modules")
                output_lines.append(f"  • Estimated {total_usages * (symbol_count / 10)} symbol usages may need review")
                output_lines.append(f"  • Test coverage recommended for {len(affected_files)} files")
                risk_level = "MEDIUM" if target_score < 0.01 else "HIGH"
            
            output_lines.append(f"\nRisk Level: {risk_level}")
            
            # Recommendations
            output_lines.append("\nRecommendations:")
            if risk_level in ["HIGH", "VERY HIGH"]:
                output_lines.append("  1. Create comprehensive tests before refactoring")
                output_lines.append("  2. Consider incremental refactoring approach")
                output_lines.append("  3. Document all API changes")
                output_lines.append("  4. Plan for gradual migration if removing")
            
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
        total_risk_score = sum(p['score'] for p in propagation_scores[:10])
        
        if total_risk_score > 1000:
            risk_level = "HIGH"
            output_lines.append(f"• Risk Level {risk_level}: HIGH - Many dependent modules likely affected")
        elif total_risk_score > 500:
            risk_level = "MEDIUM"
            output_lines.append(f"• Risk Level {risk_level}: MEDIUM - Moderate ripple effects expected")
        else:
            risk_level = "LOW"
            output_lines.append("• Risk Level: LOW - Limited propagation expected")
        
        output_lines.append(f"• Estimated modules affected: {len([p for p in propagation_scores if p['probability'] > 0.3])}")
        output_lines.append(f"• Recommended: Review and test the top {min(10, len(propagation_scores))} affected modules")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in change_propagation_analysis: {str(e)}"


# helper function not exposed as a tool
def path_to_module_fqn(file_path: str, repo_path: str) -> Optional[str]:
    """Convert file path to module FQN - reuse from coderank.py"""
    from coderank import path_to_module_fqn as _path_to_module_fqn
    return _path_to_module_fqn(file_path, repo_path)
