from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict
from sklearn.metrics import mean_squared_error

from ml4s.abstasks.AbsTask import AbsTask, ScoresDict
from ml4s.abstasks.TaskMetadata import DescriptiveStatistics, TaskMetadata

logger = logging.getLogger(__name__)


class AbsTaskEditing(AbsTask):
    """Abstract base class for SVG editing tasks in ML4S-Benchmark.
    
    Editing tasks consist of a query (input SVG) and a ground truth answer (edited SVG).
    Models generate responses which are compared with the ground truth using metrics
    like MSE or other similarity measures.
    """
    
    def _evaluate_subset(
        self,
        model: Any,
        data_split: DatasetDict | Dataset,
        **kwargs: Any,
    ) -> ScoresDict:
        """Evaluate the model on an editing task.
        
        Args:
            model: The model to evaluate
            data_split: The dataset split to evaluate on
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with evaluation scores
        """
        # Extract inputs and ground truth
        queries, ground_truths = self._get_inputs_and_targets(data_split)
        
        # Generate model responses
        responses = self._generate_responses(model, queries, **kwargs)
        
        # Calculate metrics
        scores = self._calculate_metrics(responses, ground_truths)
        
        # Add metadata
        scores["num_samples"] = len(queries)
        
        return scores
    
    def _get_inputs_and_targets(self, data_split: DatasetDict | Dataset) -> Tuple[List[str], List[str]]:
        """Extract input queries and ground truth targets from the dataset.
        
        Args:
            data_split: The dataset split to extract from
            
        Returns:
            Tuple of (queries, ground_truths)
        """
        if "query" in data_split.column_names and "ground_truth" in data_split.column_names:
            return data_split["query"], data_split["ground_truth"]
        elif "input" in data_split.column_names and "output" in data_split.column_names:
            return data_split["input"], data_split["output"]
        elif "svg_source" in data_split.column_names and "svg_target" in data_split.column_names:
            return data_split["svg_source"], data_split["svg_target"]
        else:
            column_names = data_split.column_names
            raise ValueError(
                f"Could not find input/output columns in dataset. Available columns: {column_names}"
            )
    
    def _generate_responses(
        self, 
        model: Any, 
        queries: List[str],
        **kwargs: Any,
    ) -> List[str]:
        """Generate responses from the model for the given queries.
        
        Args:
            model: The model to evaluate
            queries: List of input queries
            **kwargs: Additional arguments
            
        Returns:
            List of model-generated responses
        """
        if hasattr(model, "generate_text"):
            return model.generate_text(queries, **kwargs)
        elif hasattr(model, "generate"):
            return model.generate(queries, **kwargs)
        elif hasattr(model, "edit_svg"):
            return model.edit_svg(queries, **kwargs)
        else:
            raise NotImplementedError(
                "Model doesn't have a generate_text, generate, or edit_svg method. "
                "Override _generate_responses in your task class."
            )
    
    def _calculate_metrics(
        self, 
        responses: List[str], 
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Calculate evaluation metrics between model responses and ground truths.
        
        Args:
            responses: Model-generated responses
            ground_truths: Ground truth answers
            
        Returns:
            Dictionary with metric scores
        """
        # Calculate basic metrics
        metrics = {
            "mse": self._calculate_mse(responses, ground_truths),
        }
        
        # Add task-specific metrics
        task_metrics = self._calculate_task_specific_metrics(responses, ground_truths)
        metrics.update(task_metrics)
        
        return metrics
    
    def _calculate_mse(self, responses: List[str], ground_truths: List[str]) -> float:
        """Calculate Mean Squared Error between vector representations of responses and ground truths.
        
        Args:
            responses: Model-generated responses
            ground_truths: Ground truth answers
            
        Returns:
            MSE score
        """
        # Convert text to vector representations (to be implemented by subclasses)
        response_vectors = self._convert_to_vectors(responses)
        ground_truth_vectors = self._convert_to_vectors(ground_truths)
        
        # Calculate MSE
        mse = mean_squared_error(ground_truth_vectors, response_vectors)
        return float(mse)
    
    def _convert_to_vectors(self, texts: List[str]) -> np.ndarray:
        """Convert text responses to vector representations for distance calculations.
        
        Args:
            texts: List of text responses
            
        Returns:
            NumPy array of vector representations
        """
        # This is a placeholder that should be implemented by subclasses
        # based on the specific representation needed for the editing task
        raise NotImplementedError(
            "Method _convert_to_vectors must be implemented by subclasses."
        )
    
    def _calculate_task_specific_metrics(
        self, 
        responses: List[str], 
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Calculate task-specific metrics between responses and ground truths.
        
        Args:
            responses: Model-generated responses
            ground_truths: Ground truth answers
            
        Returns:
            Dictionary with additional metric scores
        """
        # This is a placeholder that should be implemented by subclasses
        # to add task-specific metrics beyond basic MSE
        return {}
    
    def _calculate_metrics_from_split(
        self, 
        split: str, 
        compute_overall: bool = False,
    ) -> DescriptiveStatistics:
        """Calculate descriptive statistics for a dataset split.
        
        Args:
            split: Dataset split
            compute_overall: Whether to compute overall statistics
            
        Returns:
            Dictionary with descriptive statistics
        """
        if not self.data_loaded:
            self.load_data()
            
        data = self.dataset[split]
        
        # Count samples
        stats = DescriptiveStatistics()
        stats["num_samples"] = len(data)
        
        # Extract queries and ground truths
        queries, ground_truths = self._get_inputs_and_targets(data)
        
        # Calculate query statistics
        stats["avg_query_length"] = float(np.mean([len(q) for q in queries]))
        stats["avg_query_tokens"] = float(np.mean([len(q.split()) for q in queries]))
        
        # Calculate ground truth statistics
        stats["avg_ground_truth_length"] = float(np.mean([len(gt) for gt in ground_truths]))
        stats["avg_ground_truth_tokens"] = float(np.mean([len(gt.split()) for gt in ground_truths]))
        
        # Calculate edit distance statistics if available
        if hasattr(self, "_calculate_edit_distance"):
            distances = [self._calculate_edit_distance(q, gt) for q, gt in zip(queries, ground_truths)]
            stats["avg_edit_distance"] = float(np.mean(distances))
            stats["max_edit_distance"] = float(np.max(distances))
            stats["min_edit_distance"] = float(np.min(distances))
            
        return stats 