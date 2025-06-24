
from datetime import datetime, timedelta
from typing import Dict, Set, List, Tuple
import hashlib


"""
1. **`analyze_recent_changes`**  - Main tool that:
   - Analyzes commits from the last N days
   - Ranks changes by combining CodeRank scores with commit frequency, contributor count, and change size
   - Identifies the most important recent modifications

2. **`get_commit_hotspots`**  - Finds modules frequently changed together:
   - Identifies hidden dependencies through co-change patterns
   - Helps spot potential architectural issues
   - Suggests refactoring opportunities

3. **`contributor_impact_analysis`**  - Analyzes developer patterns:
   - Tracks which developers work on critical modules
   - Identifies domain experts based on commit patterns
   - Measures contributor impact weighted by module importance

4. **`change_propagation_analysis`**  - Predicts change ripple effects:
   - Uses historical data to predict which modules will need changes
   - Identifies test files that typically change with a module
   - Provides risk assessment for proposed changes

"""


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
        module_changes = defaultdict(lambda: {
            'commits': [],
            'authors': set(),
            'commit_count': 0,
            'lines_changed': 0,
            'coderank_score': 0,
            'files': set()
        })
        
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
            
            change_scores.append({
                'module': module_fqn,
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
        output_lines.append(f"{'Module'.ljust(40)} | {'Impact'.rjust(7)} | {'CodeRank'.rjust(8)} | {'Commits'.rjust(7)} | {'Authors'.rjust(7)} | {'Lines ±'.rjust(8)}")
        output_lines.append("-" * 95)
        
        for change in top_changes:
            output_lines.append(
                f"{change['module'][:40].ljust(40)} | "
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
        
        # Find hotspots (many commits)
        hotspots = [c for c in top_changes if c['commit_count'] > 10]
        if hotspots:
            output_lines.append(f"\n• Change Hotspots (frequently modified):")
            for change in hotspots[:3]:
                output_lines.append(f"  - {change['module']}: {change['commit_count']} commits")
        
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
        cochange_pairs = defaultdict(lambda: {
            'count': 0,
            'commits': [],
            'authors': set(),
            'combined_coderank': 0
        })
        
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
        contributor_data = defaultdict(lambda: {
            'commits': 0,
            'modules_touched': set(),
            'impact_score': 0,
            'lines_added': 0,
            'lines_removed': 0,
            'important_module_commits': 0,
            'recent_commits': []
        })
        
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
        propagation_data = defaultdict(lambda: {
            'co_change_count': 0,
            'commits': [],
            'is_import_related': False,
            'is_test': False,
            'module_score': 0
        })
        
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
            output_lines.append("• Risk Level: HIGH - Many dependent modules likely affected")
        elif total_risk_score > 500:
            risk_level = "MEDIUM"
            output_lines.append("• Risk Level: MEDIUM - Moderate ripple effects expected")
        else:
            risk_level = "LOW"
            output_lines.append("• Risk Level: LOW - Limited propagation expected")
        
        output_lines.append(f"• Estimated modules affected: {len([p for p in propagation_scores if p['probability'] > 0.3])}")
        output_lines.append(f"• Recommended: Review and test the top {min(10, len(propagation_scores))} affected modules")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"Error in change_propagation_analysis: {str(e)}"


# Add a helper function that's not exposed as a tool
def path_to_module_fqn(file_path: str, repo_path: str) -> Optional[str]:
    """Convert file path to module FQN - reuse from coderank.py"""
    from coderank import path_to_module_fqn as _path_to_module_fqn
    return _path_to_module_fqn(file_path, repo_path)

