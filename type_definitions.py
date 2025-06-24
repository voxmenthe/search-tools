from typing import List, Set, TypedDict


# Type definitions for better type safety
class ModuleChangeData(TypedDict):
    commits: List[str]
    authors: Set[str]
    commit_count: int
    lines_changed: int
    coderank_score: float
    files: Set[str]

class CochangeData(TypedDict):
    count: int
    commits: List[str]
    authors: Set[str]
    combined_coderank: float

class ContributorData(TypedDict):
    commits: int
    modules_touched: Set[str]
    impact_score: float
    lines_added: int
    lines_removed: int
    important_module_commits: int
    recent_commits: List[str]

class PropagationData(TypedDict):
    co_change_count: int
    commits: List[str]
    is_import_related: bool
    is_test: bool
    module_score: float


def create_module_change_data() -> ModuleChangeData:
    """Factory function for module change data"""
    return {
        'commits': [],
        'authors': set(),
        'commit_count': 0,
        'lines_changed': 0,
        'coderank_score': 0,
        'files': set()
    }

def create_cochange_data() -> CochangeData:
    """Factory function for cochange data"""
    return {
        'count': 0,
        'commits': [],
        'authors': set(),
        'combined_coderank': 0
    }

def create_contributor_data() -> ContributorData:
    """Factory function for contributor data"""
    return {
        'commits': 0,
        'modules_touched': set(),
        'impact_score': 0,
        'lines_added': 0,
        'lines_removed': 0,
        'important_module_commits': 0,
        'recent_commits': []
    }

def create_propagation_data() -> PropagationData:
    """Factory function for propagation data"""
    return {
        'co_change_count': 0,
        'commits': [],
        'is_import_related': False,
        'is_test': False,
        'module_score': 0
    }
