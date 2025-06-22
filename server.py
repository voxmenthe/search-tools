import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional
from mcp.server.fastmcp import FastMCP


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
        Keep only rows whose **File** column *contains* this substring.
        None ⇒ no inclusion filter.
    file_must_not_contain : str | None
        Discard rows whose **File** column contains this substring.
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

