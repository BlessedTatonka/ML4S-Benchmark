from __future__ import annotations

import logging
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

from ml4s.abstasks.AbsTask import AbsTask, ScoresDict
from ml4s.abstasks.TaskMetadata import DescriptiveStatistics, TaskMetadata

logger = logging.getLogger(__name__)


class AbsTaskUnderstanding(AbsTask):
    """Abstract base class for understanding tasks in ML4S-Benchmark.
    
    Understanding tasks consist of a query (e.g., a multiple-choice question)
    and a model must choose from available answer options.
    """
    
    def _evaluate_subset(
        self,
        model: Any,
        data_split: DatasetDict | Dataset,
        **kwargs: Any,
    ) -> ScoresDict:
        """Evaluate the model on an understanding task.
        
        Args:
            model: The model to evaluate
            data_split: The dataset split to evaluate on
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with evaluation scores
        """
        # Extract inputs and ground truth
        queries, ground_truths, categories = self._get_inputs_and_targets(data_split)
        
        # Generate model responses
        responses = self._generate_responses(model, queries, **kwargs)
        
        # Extract answers from responses
        extracted_answers = self._extract_answers(responses)
        
        # Calculate metrics
        scores = self._calculate_metrics(extracted_answers, ground_truths, categories)
        
        # Add metadata
        scores["num_samples"] = len(queries)
        
        return scores
    
    def _get_inputs_and_targets(self, data_split: DatasetDict | Dataset) -> Tuple[List[str], List[str], List[str]]:
        """Extract input queries, ground truth answers, and categories from the dataset.
        
        Args:
            data_split: The dataset split to extract from
            
        Returns:
            Tuple of (queries, ground_truths, categories)
        """
        if all(col in data_split.column_names for col in ["Question", "Answer", "Subject", "A", "B", "C", "D"]):
            # Format multiple choice questions
            queries = [self._format_question(example) for example in data_split]
            return queries, data_split["Answer"], data_split["Subject"]
        else:
            column_names = data_split.column_names
            raise ValueError(
                f"Could not find expected columns in dataset. Available columns: {column_names}"
            )
    
    def _format_question(self, example: Dict[str, Any]) -> str:
        """Format a multiple choice question.
        
        Args:
            example: Dictionary containing question data
            
        Returns:
            Formatted question string
        """
        return f"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{example['Question']}

A) {example['A']}
B) {example['B']}
C) {example['C']}
D) {example['D']}
""".strip()
    
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
        else:
            raise NotImplementedError(
                "Model doesn't have a generate_text or generate method. "
                "Override _generate_responses in your task class."
            )
    
    def _extract_answers(self, responses: List[str]) -> List[str]:
        """Extract answers from model responses.
        
        Args:
            responses: Model-generated responses
            
        Returns:
            List of extracted answers (A, B, C, D)
        """
        answer_pattern = r"(?i)Answer\s*:\s*([A-D])"
        extracted_answers = []
        
        for response in responses:
            match = re.search(answer_pattern, response)
            extracted_answer = match.group(1).upper() if match else None
            extracted_answers.append(extracted_answer)
        
        return extracted_answers
    
    def _calculate_metrics(
        self, 
        extracted_answers: List[str], 
        ground_truths: List[str],
        categories: List[str],
    ) -> Dict[str, float]:
        """Calculate evaluation metrics between extracted answers and ground truths.
        
        Args:
            extracted_answers: Extracted model answers
            ground_truths: Ground truth answers
            categories: Question categories/subjects
            
        Returns:
            Dictionary with metric scores
        """
        # Calculate overall accuracy
        correct = [pred == true for pred, true in zip(extracted_answers, ground_truths) if pred is not None]
        accuracy = sum(correct) / len(correct) if correct else 0.0
        
        # Calculate category-specific accuracies
        category_scores = {}
        category_counts = {}
        
        for category, pred, true in zip(categories, extracted_answers, ground_truths):
            if category not in category_counts:
                category_counts[category] = 0
                category_scores[category] = 0
            
            if pred is not None:
                category_counts[category] += 1
                if pred == true:
                    category_scores[category] += 1
        
        # Calculate accuracy for each category
        metrics = {"accuracy": accuracy}
        for category, score in category_scores.items():
            count = category_counts[category]
            if count > 0:
                metrics[f"{category}_accuracy"] = score / count
        
        return metrics
    
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
        
        # Count examples per category
        if "Subject" in data.column_names:
            categories = data["Subject"]
            unique_categories = set(categories)
            for category in unique_categories:
                count = sum(1 for c in categories if c == category)
                stats[f"{category}_count"] = count
                stats[f"{category}_percent"] = 100 * count / len(categories)
        
        return stats 