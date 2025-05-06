from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DescriptiveStatistics(dict):
    """Type for descriptive statistics of a dataset."""
    pass


class TaskMetadata:
    """Metadata for a task in ML4S-Benchmark."""
    
    def __init__(
        self,
        name: str,
        description: str,
        type: str,
        dataset: Dict[str, Any],
        eval_splits: List[str] = None,
        main_score: str = "accuracy",
        modalities: List[str] = None,
        results_dir: str = "results"
    ):
        self.name = name
        self.description = description
        self.type = type
        self.dataset = dataset
        self.eval_splits = eval_splits or ["test"]
        self.main_score = main_score
        self.modalities = modalities or ["text"]
        self.results_dir = results_dir
    
    @property
    def descriptive_stat_path(self) -> Path:
        """Path to the descriptive statistics file."""
        stats_dir = Path(self.results_dir) / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        return stats_dir / f"{self.name}_stats.json"
    
    @property
    def descriptive_stats(self) -> Dict[str, Any]:
        """Load descriptive statistics from file."""
        if not self.descriptive_stat_path.exists():
            return {}
        
        with open(self.descriptive_stat_path) as f:
            return json.load(f)
    
    def __hash__(self) -> int:
        """Hash the metadata by its name."""
        return hash(self.name) 