import os
import re
import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Iterable
from collections import defaultdict
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
            text=True
        )
        
        if rg_result.returncode != 0:
            # Try without JSON for better error message
            simple_result = subprocess.run(
                ["rg", "-n", "-i", keyword, repo_path],
                capture_output=True,
                text=True
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
