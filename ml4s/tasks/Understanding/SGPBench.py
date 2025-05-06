from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from ml4s.abstasks.AbsTaskUnderstanding import AbsTaskUnderstanding
from ml4s.abstasks.TaskMetadata import TaskMetadata
from ml4s.abstasks.AbsTask import ScoresDict

logger = logging.getLogger(__name__)


class SGPBench(AbsTaskUnderstanding):
    """Understanding task for evaluating LLMs on SGP-Bench multiple choice questions.
    
    Uses the sgp-bench/sgp-bench dataset which contains multiple-choice questions
    covering various subjects with emphasis on spatial and visual understanding.
    """
    
    def __init__(self, seed: int = 42, max_samples: Optional[int] = None, **kwargs: Any):
        """Initialize the SGPBench task.
        
        Args:
            seed: Random seed for reproducibility
            max_samples: Maximum number of samples to evaluate (if None, use all)
            **kwargs: Additional arguments
        """
        # Define task metadata
        self.metadata = TaskMetadata(
            name="sgp_bench",
            description="Evaluates model's ability to answer multiple-choice questions requiring spatial understanding",
            type="Understanding",
            dataset={
                "path": "sgp-bench/sgp-bench",
                "split": "svg",  # Use 'svg' for SVG understanding task
                "revision": "main"
            },
            eval_splits=["svg"],
            main_score="accuracy",
            modalities=["text", "image"],
        )
        
        # Maximum number of samples to evaluate
        self.max_samples = max_samples
        
        # Directory to cache model responses
        self.cache_dir = None
        
        # Store dataset size before filtering
        self.original_dataset_sizes = {}
        
        # Response cache file (set later)
        self.response_cache_file = None
        
        # Initialize parent class
        super().__init__(seed=seed, **kwargs)
        
    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub with custom handling for SGP-Bench."""
        if self.data_loaded:
            return
        
        from datasets import load_dataset
        # Load dataset using the split specified in metadata
        split = self.metadata.dataset.get("split", "svg")
        self.dataset = {
            split: load_dataset(
                self.metadata.dataset["path"], 
                split=split,
                revision=self.metadata.dataset.get("revision", "main")
            )
        }
        
        # Store original dataset sizes before any transformation
        for split_name, split_data in self.dataset.items():
            self.original_dataset_sizes[split_name] = len(split_data)
            logger.info(f"Original dataset size for {split_name} split: {self.original_dataset_sizes[split_name]}")
        
        self.dataset_transform()
        self.data_loaded = True
        
    def dataset_transform(self):
        """Apply transformations to the dataset after loading.
        
        This method is called after the dataset is loaded from Hugging Face.
        It applies any necessary filtering or transformations.
        """
        # Shuffle dataset with fixed seed for reproducibility
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].shuffle(seed=self.seed)
            
            # If max_samples is specified, select only that many samples
            if self.max_samples and len(self.dataset[split]) > self.max_samples:
                self.dataset[split] = self.dataset[split].select(range(self.max_samples))
    
    def _setup_response_cache(self, model_name: str = "unknown", output_dir: str = "results"):
        """Set up cache for storing model responses.
        
        Args:
            model_name: Name of the model being evaluated
            output_dir: Directory to store results
        """
        from ml4s.run import get_cache_file_path
        
        # Get cache file path
        self.response_cache_file = get_cache_file_path(
            task_name="SGPBench",
            model_name=model_name,
            output_dir=output_dir
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.response_cache_file), exist_ok=True)
    
    def _save_response_to_jsonl(self, sample_id: str, query: str, response: str, model_name: str):
        """Save model response to the JSONL file.
        
        Args:
            sample_id: ID of the sample
            query: The input query
            response: Model's response
            model_name: Name of the model
        """
        if not self.response_cache_file:
            return
            
        # Prepare record
        record = {
            "sample_id": sample_id,
            "query": query,
            "response": response,
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "retry_count": 0
        }
        
        # Check if sample_id already exists in the file
        existing_entries = {}
        
        # Read existing entries if file exists
        if os.path.exists(self.response_cache_file):
            try:
                with open(self.response_cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if "sample_id" in entry:
                                existing_entries[entry["sample_id"]] = True
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Error reading response cache: {e}")
        
        # Only write if sample_id is not already in the file
        if sample_id not in existing_entries:
            with open(self.response_cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        else:
            logger.debug(f"Skipping duplicate response for sample_id: {sample_id}")
    
    def process_dataset_with_model(self, model, dataset, split, **kwargs):
        """Process the dataset with the model and calculate metrics.
        
        Args:
            model: The model to evaluate
            dataset: The dataset dictionary
            split: The split to use
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with evaluation scores
        """
        # Store the current split for reference in metrics calculation
        self.current_evaluation_split = split
        
        # Get dataset split
        data_split = dataset[split]
        
        # Get model name for caching responses
        model_name = kwargs.get("model_name", getattr(model, "model_name", "unknown"))
        
        # Set up response cache
        self._setup_response_cache(model_name=model_name)
        
        # Extract queries, ground truths, and sample IDs
        queries, ground_truths, categories, sample_ids = self._get_inputs_and_targets_with_ids(data_split)
        
        # Generate responses
        logger.info(f"Generating responses for {len(queries)} queries")
        
        # Get the global progress bar if provided
        global_progress_bar = kwargs.pop('global_progress_bar', None)
        
        # Generate responses and pass through the progress bar
        responses = model.generate_text(
            queries, 
            sample_ids=sample_ids, 
            global_progress_bar=global_progress_bar,
            **kwargs
        )
        
        # Save responses to JSONL file
        for sample_id, query, response in zip(sample_ids, queries, responses):
            self._save_response_to_jsonl(sample_id, query, response, model_name)
        
        # Extract answers from responses
        extracted_answers = self._extract_answers(responses)
        
        # Calculate metrics
        logger.info("Calculating metrics")
        scores = self._calculate_metrics(extracted_answers, ground_truths, categories)
        
        # Add metadata
        scores["num_samples"] = len(queries)
        
        return scores
    
    def _get_inputs_and_targets_with_ids(self, data_split) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract input queries, ground truth answers, categories, and sample IDs from the dataset.
        
        Args:
            data_split: The dataset split
            
        Returns:
            Tuple of (queries, ground_truths, categories, sample_ids)
        """
        # Get queries and answers from the dataset
        queries, ground_truths, categories = self._get_inputs_and_targets(data_split)
        
        # Generate sample IDs if not present in the dataset
        if "id" in data_split.column_names:
            sample_ids = data_split["id"]
        else:
            # Generate simple IDs based on index
            sample_ids = [f"{i}_{hashlib.md5(q.encode()).hexdigest()[:5]}" for i, q in enumerate(queries)]
        
        return queries, ground_truths, categories, sample_ids
    
    def _calculate_metrics(
        self, 
        extracted_answers: List[str], 
        ground_truths: List[str],
        categories: List[str],
    ) -> Dict[str, float]:
        """Calculate metrics for the task.
        
        Args:
            extracted_answers: Extracted answers from model responses
            ground_truths: Ground truth answers
            categories: Question categories/subjects
            
        Returns:
            Dictionary with calculated metrics
        """
        # Get base metrics from parent class
        metrics = super()._calculate_metrics(extracted_answers, ground_truths, categories)
        
        # Get original dataset size for the current split
        current_split = getattr(self, "current_evaluation_split", "svg")
        total_dataset_size = self.original_dataset_sizes.get(current_split, len(extracted_answers))
        
        # Record total samples from original dataset
        metrics["total_samples"] = total_dataset_size
        
        # Record evaluated samples separately
        metrics["evaluated_samples"] = len(extracted_answers)
        
        # Count valid responses (where we could extract an answer)
        valid_responses = sum(1 for ans in extracted_answers if ans is not None)
        metrics["valid_responses"] = valid_responses
        metrics["valid_response_rate"] = valid_responses / len(extracted_answers) if len(extracted_answers) > 0 else 0
        
        # Calculate confidence based on valid responses relative to original dataset size
        if total_dataset_size > 0:
            # Simple confidence calculation
            metrics["confidence"] = valid_responses / total_dataset_size
        else:
            metrics["confidence"] = 0.0
        
        # Set the primary metric as the main score
        metrics["primary_metric"] = metrics["accuracy"]
        
        return metrics 

    def evaluate(
        self,
        model: Any,
        split: str = "svg",  # Default to 'svg' instead of 'test'
        **kwargs: Any,
    ) -> ScoresDict:
        """Evaluates a model on the SGPBench task.

        Args:
            model: The model to evaluate
            split: Which datasplit to be used, defaults to 'svg'
            **kwargs: Additional arguments passed to _evaluate_subset
        """
        if not self.data_loaded:
            self.load_data()

        # Ensure we're using a valid split 
        if split not in self.dataset:
            # Fall back to the first available split
            logger.warning(f"Split '{split}' not found in dataset. Using '{list(self.dataset.keys())[0]}' instead.")
            split = list(self.dataset.keys())[0]
            
        # Store the current split for reference in metrics calculation
        self.current_evaluation_split = split

        # Get model name for caching responses
        model_name = kwargs.get("model_name", getattr(model, "model_name", "unknown"))
        
        # Set up response cache
        self._setup_response_cache(model_name=model_name)

        logger.info(f"Task: {self.metadata.name}, split: {split}. Running...")
        data_split = self.dataset[split]
        
        # Use process_dataset_with_model if available for proper response logging
        if hasattr(self, 'process_dataset_with_model'):
            scores = self.process_dataset_with_model(model, self.dataset, split, **kwargs)
        else:
            scores = self._evaluate_subset(model, data_split, **kwargs)
        
        return scores 