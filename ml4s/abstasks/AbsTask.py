from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from ml4s.abstasks.TaskMetadata import DescriptiveStatistics, TaskMetadata

logger = logging.getLogger(__name__)

ScoresDict = Dict[str, Any]


class AbsTask(ABC):
    """Abstract base class for all tasks in ML4S-Benchmark."""
    
    metadata: TaskMetadata
    abstask_prompt: Optional[str] = None
    _eval_splits: Optional[List[str]] = None
    dataset: Optional[DatasetDict] = None
    data_loaded: bool = False

    def __init__(self, seed: int = 42, **kwargs: Any):
        """Initialize the task with a random seed.
        
        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
        """
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def dataset_transform(self):
        """Transform operations applied to the dataset after loading.
        Override this method if your dataset requires any transformation.
        """
        pass

    def evaluate(
        self,
        model: Any,
        split: str = "test",
        **kwargs: Any,
    ) -> ScoresDict:
        """Evaluates a model on the task.

        Args:
            model: The model to evaluate
            split: Which datasplit to be used
            **kwargs: Additional arguments passed to _evaluate_subset
        """
        if not self.data_loaded:
            self.load_data()

        logger.info(f"Task: {self.metadata.name}, split: {split}. Running...")
        data_split = self.dataset[split]
        scores = self._evaluate_subset(model, data_split, **kwargs)
        
        return scores

    @abstractmethod
    def _evaluate_subset(
        self,
        model: Any,
        data_split: DatasetDict | Dataset,
        **kwargs: Any,
    ) -> ScoresDict:
        """Evaluate the model on a specific subset of the data.
        
        Args:
            model: The model to evaluate
            data_split: The dataset split to evaluate on
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with evaluation scores
        """
        raise NotImplementedError(
            "You must implement _evaluate_subset method in your task class."
        )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        
        from datasets import load_dataset
        self.dataset = load_dataset(**self.metadata.dataset)
        self.dataset_transform()
        self.data_loaded = True

    @abstractmethod
    def _calculate_metrics_from_split(
        self, split: str, compute_overall: bool = False
    ) -> DescriptiveStatistics:
        """Calculate descriptive statistics for a dataset split.
        
        Args:
            split: Dataset split
            compute_overall: Whether to compute overall statistics
            
        Returns:
            Dictionary with descriptive statistics
        """
        raise NotImplementedError

    def filter_eval_splits(self, eval_splits: Optional[List[str]]) -> AbsTask:
        """Filter the evaluation splits of the task."""
        self._eval_splits = eval_splits
        return self

    @property
    def is_aggregate(self) -> bool:
        """Whether the task is aggregate."""
        return False

    @property
    def eval_splits(self) -> List[str]:
        """Get evaluation splits to use."""
        if self._eval_splits:
            return self._eval_splits
        return self.metadata.eval_splits

    @property
    def modalities(self) -> List[str]:
        """Returns the modalities of the task"""
        return self.metadata.modalities

    def __repr__(self) -> str:
        """Format the representation of the task."""
        return f"{self.__class__.__name__}(name='{self.metadata.name}')"

    def __hash__(self) -> int:
        """Hash the task by its metadata."""
        return hash(self.metadata)
