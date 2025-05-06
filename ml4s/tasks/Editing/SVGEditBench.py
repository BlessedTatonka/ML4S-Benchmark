from __future__ import annotations

import json
import re
import logging
import os
import tempfile
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import io
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
import cairosvg
from PIL import Image

from ml4s.abstasks.AbsTaskEditing import AbsTaskEditing
from ml4s.abstasks.TaskMetadata import TaskMetadata

# Set logging level to reduce warnings
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Only show INFO and above, not DEBUG


class SVGEditBench(AbsTaskEditing):
    """SVG Editing task for evaluating LLMs on SVG editing capabilities.
    
    Uses the VectorGraphics/svg_edit_bench dataset which contains pairs of
    SVG queries and answers.
    
    Notes on SVG rasterization:
    - SVGs are converted to PNG with a white background for transparent areas
    - Pixel values are standardized to fall between 0 and 1 (normalized)
    - MSE is calculated across all color channels and averaged, resulting in
      MSE values also between 0 and 1
    """
    
    # Task types in the dataset with proper naming for metrics
    TASK_TYPES = [
        "color_change", 
        "position_change", 
        "attribute_change", 
        "shape_addition", 
        "shape_deletion",
        "compression"
    ]
    
    # Map from task type internal names to pretty names for metrics
    TASK_TYPE_METRIC_NAMES = {
        # Original mapping based on text extraction
        "color_change": "ChangeColor",
        "position_change": "ChangePosition", 
        "attribute_change": "ChangeAttribute",
        "shape_addition": "AddShape",
        "shape_deletion": "DeleteShape",
        "compression": "Compression",
        "unknown": "Unknown",
        
        # Dataset-specific task type names
        "ChangeColor": "ChangeColor",
        "ChangePosition": "ChangePosition",
        "ChangeAttribute": "ChangeAttribute",
        "AddShape": "AddShape",
        "DeleteShape": "DeleteShape",
        "Compression": "Compression",
        
        # Handle potential lowercase variations in the dataset
        "changecolor": "ChangeColor",
        "changeposition": "ChangePosition",
        "changeattribute": "ChangeAttribute",
        "addshape": "AddShape",
        "deleteshape": "DeleteShape",
        "compression": "Compression",
    }
    
    def __init__(self, seed: int = 42, penalize_non_rasterizable: bool = True, raster_size: int = 72, 
                 skip_empty_responses: bool = False, **kwargs: Any):
        # Define task metadata
        self.metadata = TaskMetadata(
            name="svg_edit_bench",
            description="Evaluates model's ability to edit SVG files based on instructions",
            type="Editing",
            dataset={
                "path": "VectorGraphics/svg_edit_bench",
                "revision": "main"
            },
            eval_splits=["test"],
            main_score="primary_metric",  # Changed from "mean_score" to "primary_metric"
            modalities=["svg", "text"],
        )
        
        # Flag to determine how to handle non-rasterizable SVGs
        self.penalize_non_rasterizable = penalize_non_rasterizable
        
        # Size of rasterized PNG (as specified in the paper)
        self.raster_size = raster_size
        
        # Flag to skip calculation for empty responses
        self.skip_empty_responses = skip_empty_responses
        
        # Store dataset size before filtering
        self.original_dataset_sizes = {}
        
        # Response cache file (set later)
        self.response_cache_file = None
        
        # Initialize parent class
        super().__init__(seed=seed, **kwargs)
    
    def _get_inputs_and_targets(self, data_split) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract input queries, ground truth targets, task types, and IDs from the dataset.
        
        Args:
            data_split: Dataset split
            
        Returns:
            Tuple of (queries, answers, task_types, sample_ids)
        """
        # Extract queries and answers first
        if "query" in data_split.column_names and "answer" in data_split.column_names:
            queries = data_split["query"]
            answers = data_split["answer"]
            
            # Log column information for debugging
            logger.info(f"Found query and answer columns in dataset")
            logger.info(f"Dataset has {len(queries)} examples")
            
            # Extract task types if available
            task_types = None
            if "task_type" in data_split.column_names:
                task_types = data_split["task_type"]
                logger.info(f"Found task_type column in dataset")
                logger.info(f"Task types found: {set(task_types)}")
            
            # Extract or generate sample IDs
            sample_ids = None
            if "id" in data_split.column_names:
                sample_ids = data_split["id"]
                logger.info(f"Using {len(sample_ids)} sample IDs from dataset")
            else:
                sample_ids = [f"sample_{i}" for i in range(len(queries))]
                logger.info(f"Generated {len(sample_ids)} sample IDs")
            
            return queries, answers, task_types, sample_ids
        else:
            # Fall back to parent implementation or try other column combinations
            logger.warning(f"Could not find 'query' and 'answer' columns. Available columns: {data_split.column_names}")
            return super()._get_inputs_and_targets(data_split)
    
    def _generate_responses(self, model, queries, **kwargs):
        """Generate responses from the model for the given queries.
        
        Adds specific prompting for SVG editing and extracts just the SVG code
        from model responses.
        """
        prompt_template = """
You are an expert in SVG editing. Your task is to modify the provided SVG code according to the instructions.
Return ONLY the modified SVG code, nothing else.

{query}

Modified SVG:
"""
        
        # Add the prompt template to each query
        prompted_queries = [prompt_template.format(query=query) for query in queries]
        
        # Generate responses
        raw_responses = super()._generate_responses(model, prompted_queries, **kwargs)
        
        # Extract SVG code from responses
        cleaned_responses = []
        for response in raw_responses:
            # Try to extract just the SVG code using regex
            svg_match = re.search(r'<svg[^>]*>.*?</svg>', response, re.DOTALL)
            if svg_match:
                cleaned_responses.append(svg_match.group(0))
            else:
                # If no SVG tag found, use the raw response
                cleaned_responses.append(response)
                
        return cleaned_responses
    
    def _clean_svg_text(self, svg_text: str) -> str:
        """Clean SVG text by removing markdown code block delimiters.
        
        Args:
            svg_text: Raw SVG text potentially containing markdown code blocks
            
        Returns:
            Cleaned SVG code
        """
        if not svg_text:
            return ""
            
        # Remove markdown code block delimiters
        cleaned_text = svg_text
        
        # Remove ```svg opening and ``` closing tags
        cleaned_text = re.sub(r'```svg\s*', '', cleaned_text)
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
        
        # Also handle cases where it's just ``` without svg
        cleaned_text = re.sub(r'```\s*', '', cleaned_text)
        
        # Make sure we have the SVG tags
        if "<svg" not in cleaned_text:
            # Log at debug level instead of warning to reduce noise
            logger.debug(f"No SVG tag found after cleaning: {cleaned_text[:50]}...")
            return cleaned_text
            
        # Extract just the SVG content if there's other text
        svg_match = re.search(r'<svg[^>]*>.*?</svg>', cleaned_text, re.DOTALL)
        if svg_match:
            return svg_match.group(0)
            
        return cleaned_text
    
    def _is_rasterizable(self, svg_text: str) -> bool:
        """Check if an SVG can be rasterized using cairosvg.
        
        Args:
            svg_text: SVG code as a string
            
        Returns:
            True if rasterizable, False otherwise
        """
        # Clean SVG text first
        cleaned_svg = self._clean_svg_text(svg_text)
        
        try:
            # Try to convert to PNG in memory with white background
            cairosvg.svg2png(
                bytestring=cleaned_svg.encode('utf-8'),
                background_color="white"  # Set white background for transparent areas
            )
            return True
        except Exception as e:
            logger.debug(f"SVG rasterization failed: {e}")
            return False
    
    def _rasterize_svg(self, svg_text: str) -> Optional[np.ndarray]:
        """Rasterize SVG to a NumPy array for pixel-level comparison.
        
        Args:
            svg_text: SVG code as a string
            
        Returns:
            NumPy array of pixel values, or None if rasterization failed
        """
        # Make sure we have cleaned SVG text
        svg_text = self._clean_svg_text(svg_text)
        
        if not svg_text or not ("<svg" in svg_text and "</svg>" in svg_text):
            # Log at debug level instead of warning to reduce noise
            logger.debug("Invalid SVG format for rasterization")
            return None
            
        try:
            # Convert to PNG in memory with white background
            # The background parameter ensures that transparent areas are rendered as white
            png_data = cairosvg.svg2png(
                bytestring=svg_text.encode('utf-8'),
                output_width=self.raster_size,
                output_height=self.raster_size,
                background_color="white"  # Set white background for transparent areas
            )
            
            # Open image from binary data and convert to numpy array
            img = Image.open(io.BytesIO(png_data))
            img_array = np.array(img)
            
            # Normalize to [0, 1] range for all channels
            img_array = img_array.astype(float) / 255.0
            
            # Add a small check to ensure the array has content
            if img_array.size == 0:
                logger.debug("Rasterization produced empty array")
                return None
                
            return img_array
        except Exception as e:
            # Log at debug level instead of warning to reduce noise
            logger.debug(f"SVG rasterization failed: {e}")
            return None
    
    def _get_rasterized_size(self, svg_text: str) -> int:
        """Get the size in bytes of the rasterized SVG.
        
        Args:
            svg_text: SVG code as a string
            
        Returns:
            Size in bytes of the rasterized PNG, or -1 if not rasterizable
        """
        # Clean SVG text first
        cleaned_svg = self._clean_svg_text(svg_text)
        
        try:
            # Convert to PNG in memory with white background
            png_data = cairosvg.svg2png(
                bytestring=cleaned_svg.encode('utf-8'),
                background_color="white"  # Set white background for transparent areas
            )
            return len(png_data)
        except Exception as e:
            logger.debug(f"SVG rasterization failed: {e}")
            return -1
    
    def _extract_task_type(self, query: str) -> str:
        """Extract the task type from the query.
        
        Args:
            query: The query string from the dataset
            
        Returns:
            One of the task types or "unknown"
        """
        # Look for task type keywords in the query
        for task_type in self.TASK_TYPES:
            # Replace underscore with space for matching
            search_term = task_type.replace('_', ' ')
            if search_term in query.lower():
                return task_type
        
        # Default to unknown if no match found
        return "unknown"
    
    def _calculate_mse_from_raster(self, response: str, ground_truth: str) -> Tuple[float, bool]:
        """Calculate MSE between rasterized SVGs.
        
        Notes:
            - SVGs are rasterized with a white background for transparent areas
            - Pixel values are normalized to range [0, 1]
            - MSE is calculated across all color channels and averaged
            - This produces MSE values in the range [0, 1]
        
        Args:
            response: Model-generated SVG
            ground_truth: Ground truth SVG
            
        Returns:
            Tuple of (MSE score, success flag)
        """
        # Clean SVGs first
        cleaned_response = self._clean_svg_text(response)
        cleaned_ground_truth = self._clean_svg_text(ground_truth)
        
        # Skip empty responses if configured
        if self.skip_empty_responses and (not cleaned_response or cleaned_response.strip() == ""):
            return 0.0, False
            
        # High MSE value for non-rasterizable SVGs if penalizing
        high_mse = 1.0  # Using 1.0 as the maximum possible MSE in our normalized scale
        
        # Rasterize SVGs
        response_raster = self._rasterize_svg(cleaned_response)
        gt_raster = self._rasterize_svg(cleaned_ground_truth)
        
        # Check if both rasterized successfully
        if response_raster is not None and gt_raster is not None:
            # Calculate MSE between raster images
            try:
                # Ensure same shape by using min dimensions
                min_h = min(response_raster.shape[0], gt_raster.shape[0])
                min_w = min(response_raster.shape[1], gt_raster.shape[1])
                min_c = min(response_raster.shape[2], gt_raster.shape[2])
                
                # Crop to same dimensions
                response_crop = response_raster[:min_h, :min_w, :min_c]
                gt_crop = gt_raster[:min_h, :min_w, :min_c]
                
                # Check if crops are identical (potential issue)
                if np.array_equal(response_crop, gt_crop):
                    # Calculate byte-level difference as a fallback check
                    if cleaned_response.strip() != cleaned_ground_truth.strip():
                        # Log at debug level instead of warning
                        logger.debug("Images are identical but SVG code differs - using minimum MSE")
                        return 0.001, True  # Use small non-zero value to indicate difference
                
                # Calculate pixel-wise MSE with small epsilon to avoid exact zero
                # Since we're using normalized [0, 1] values, this will produce MSE in range [0, 1]
                mse = np.mean((response_crop - gt_crop) ** 2)
                
                # Ensure MSE is not exactly zero if SVGs differ
                if mse < 0.0001 and cleaned_response.strip() != cleaned_ground_truth.strip():
                    mse = 0.0001  # Minimum non-zero MSE for different SVGs
                    
                return float(mse), True
            except Exception as e:
                # Log at debug level instead of warning
                logger.debug(f"Error calculating MSE from rasters: {e}")
                # Fall back to high MSE
                if self.penalize_non_rasterizable:
                    return high_mse, False
        
        # Handle non-rasterizable SVGs
        if self.penalize_non_rasterizable:
            return high_mse, False
        else:
            # Use a small non-zero value instead of zero to indicate failure
            return 0.5, False
    
    def _is_valid_svg(self, svg_text: str) -> bool:
        """Check if the text is a valid SVG.
        
        Args:
            svg_text: The SVG text to validate
            
        Returns:
            True if the SVG is valid, False otherwise
        """
        # Clean the SVG text first
        cleaned_svg = self._clean_svg_text(svg_text)
        
        # Check if it's empty or None
        if not cleaned_svg or cleaned_svg.strip() == "":
            # Skip empty responses if configured
            return not self.skip_empty_responses
            
        # Check if it contains basic SVG elements
        if not ("<svg" in cleaned_svg and "</svg>" in cleaned_svg):
            return False
            
        return True
        
    def _get_filtered_data(self, responses, ground_truths, queries=None):
        """Filter out invalid responses and return valid data.
        
        Args:
            responses: List of model responses
            ground_truths: List of ground truth SVGs
            queries: List of query strings (optional)
            
        Returns:
            Tuple of (filtered_responses, filtered_ground_truths, filtered_queries, valid_indices)
        """
        filtered_responses = []
        filtered_ground_truths = []
        filtered_queries = [] if queries else None
        valid_indices = []
        
        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            # Skip empty responses if configured
            if self.skip_empty_responses and (not response or response.strip() == ""):
                continue
                
            # Check if both response and ground truth are valid
            if self._is_valid_svg(response) and self._is_valid_svg(ground_truth):
                filtered_responses.append(response)
                filtered_ground_truths.append(ground_truth)
                if queries is not None:
                    filtered_queries.append(queries[i])
                valid_indices.append(i)
            else:
                logger.debug(f"Skipping invalid SVG at index {i}")
        
        return filtered_responses, filtered_ground_truths, filtered_queries, valid_indices
    
    def _calculate_confidence_metrics(self, valid_count, total_count):
        """Calculate confidence metrics based on the ratio of valid samples.
        
        This implements a confidence metric with the following properties:
        1. Confidence = 1.0 when all samples are valid
        2. Confidence approaches 0 as the proportion of valid samples decreases
        3. Small drops in valid samples are penalized less than large drops
        4. Uses a sigmoid-based function for a more nuanced measure
        
        Args:
            valid_count: Number of valid samples
            total_count: Total number of samples
            
        Returns:
            Dictionary of confidence metrics
        """
        metrics = {}
        
        # Simple confidence = valid/total (linear relationship)
        simple_confidence = valid_count / total_count if total_count > 0 else 0.0
        
        # Calculate advanced confidence using a sigmoid function
        # This creates a more nuanced measure that:
        # - Stays close to 1.0 when the vast majority of samples are valid
        # - Drops more rapidly as the proportion of valid samples decreases
        # - Approaches 0 when very few samples are valid
        if total_count > 0:
            # Ratio of valid to total
            ratio = valid_count / total_count
            
            # Parameters for the sigmoid function
            k = 12  # Steepness of the sigmoid curve
            midpoint = 0.7  # Point at which confidence drops more rapidly
            
            # Apply sigmoid transformation to ratio
            # This maps the range [0,1] to [0,1] with a nonlinear relationship
            sigmoid_factor = 1.0 / (1.0 + np.exp(-k * (ratio - midpoint)))
            
            # Normalize to ensure 1.0 when ratio = 1.0
            normalized_sigmoid = sigmoid_factor / (1.0 / (1.0 + np.exp(-k * (1.0 - midpoint))))
            
            # Combine linear and sigmoid components
            advanced_confidence = (0.5 * ratio) + (0.5 * normalized_sigmoid)
        else:
            advanced_confidence = 0.0
            
        # Calculate uncertainty as complementary to confidence
        uncertainty = 1.0 - advanced_confidence
        
        # Add metrics to result
        metrics["simple_confidence"] = simple_confidence
        metrics["confidence"] = advanced_confidence
        metrics["uncertainty"] = uncertainty
        
        return metrics
    
    def _check_rasterizable_data(self, responses, ground_truths):
        """Check which SVG pairs can be rasterized properly.
        
        Args:
            responses: List of response SVGs
            ground_truths: List of ground truth SVGs
            
        Returns:
            Tuple of (rasterizable_responses, rasterizable_ground_truths, rasterizable_indices)
        """
        rasterizable_responses = []
        rasterizable_ground_truths = []
        rasterizable_indices = []
        
        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            # Clean SVGs first
            cleaned_response = self._clean_svg_text(response)
            cleaned_ground_truth = self._clean_svg_text(ground_truth)
            
            # Check if both can be rasterized
            if self._is_rasterizable(cleaned_response) and self._is_rasterizable(cleaned_ground_truth):
                rasterizable_responses.append(response)
                rasterizable_ground_truths.append(ground_truth)
                rasterizable_indices.append(i)
        
        return rasterizable_responses, rasterizable_ground_truths, rasterizable_indices
    
    def _calculate_metrics(self, responses, ground_truths, queries=None, task_types=None):
        """Calculate evaluation metrics between model responses and ground truths.
        
        Args:
            responses: Model-generated responses
            ground_truths: Ground truth answers
            queries: Original queries (optional, used for task type extraction if task_types not provided)
            task_types: Task types directly from the dataset (preferred over extraction)
            
        Returns:
            Dictionary with metric scores
        """
        # Initialize with only the essential metrics
        metrics = {}
        
        # Use the original dataset size for total_samples, not the filtered sample count
        current_split = getattr(self, "current_evaluation_split", "test")
        total_dataset_size = self.original_dataset_sizes.get(current_split, len(responses))
        
        # Record total samples from the original dataset
        metrics["total_samples"] = total_dataset_size
        
        # Record evaluated samples separately
        metrics["evaluated_samples"] = len(responses)
        
        # Filter out invalid SVGs
        filtered_responses, filtered_ground_truths, filtered_queries, valid_indices = self._get_filtered_data(
            responses, ground_truths, queries
        )
        
        # Record valid responses count
        metrics["valid_responses"] = len(filtered_responses)
        
        # Filter task_types based on valid indices if available
        filtered_task_types = None
        if task_types is not None:
            filtered_task_types = [task_types[i] for i in valid_indices]
        
        # Check which of the valid responses are rasterizable
        rasterizable_responses, rasterizable_ground_truths, rasterizable_indices = self._check_rasterizable_data(
            filtered_responses, filtered_ground_truths
        )
        
        # Record rasterizable responses count
        metrics["rasterizable_responses"] = len(rasterizable_responses)
        
        # Calculate confidence metrics based on rasterizable responses relative to the original dataset size
        rasterizable_confidence_metrics = self._calculate_confidence_metrics(
            len(rasterizable_responses), total_dataset_size
        )
        metrics["confidence"] = rasterizable_confidence_metrics["confidence"]
        
        # If all samples are invalid, return early with high MSE
        if len(filtered_responses) == 0:
            logger.info("No valid SVG pairs found for evaluation")
            high_mse = 1.0
            metrics["mse"] = high_mse
            metrics["primary_metric"] = high_mse
            return metrics
        
        # Calculate overall MSE for all samples
        total_mse = 0.0
        total_count = 0
        
        for i, (response, ground_truth) in enumerate(zip(filtered_responses, filtered_ground_truths)):
            mse, success = self._calculate_mse_from_raster(response, ground_truth)
            total_mse += mse
            total_count += 1
            
            # Debug output for the first few items (using debug level)
            if i < 5:  # Only log first 5 to avoid flooding
                logger.debug(f"Sample {i} MSE: {mse}, Success: {success}")
        
        if total_count > 0:
            metrics["mse"] = total_mse / total_count
        else:
            metrics["mse"] = 0.0
        
        # If we don't have task types, try to extract them from queries
        if filtered_task_types is None and filtered_queries is not None:
            # Extract from queries as a fallback
            logger.info("No task_type column found, extracting task types from queries")
            filtered_task_types = [self._extract_task_type(query) for query in filtered_queries]
        
        # If we still don't have task types, return early
        if filtered_task_types is None:
            logger.info("No task types available, cannot calculate task-specific metrics")
            metrics["primary_metric"] = metrics["mse"]
            return metrics
        
        # Track task-type specific valid/invalid counts for confidence metrics
        task_type_original_counts = defaultdict(int)
        
        # If we have the original task_types (before filtering), count them
        if task_types is not None:
            for task_type in task_types:
                if task_type:  # Skip empty task types
                    task_type_original_counts[task_type] += 1
        
        # Now we have task types (either from dataset or extracted), calculate metrics per task
        # Group examples by task type
        task_type_indices = defaultdict(list)
        for i, task_type in enumerate(filtered_task_types):
            task_type_indices[task_type].append(i)
        
        # Log task type distribution
        logger.info(f"Task type distribution: {dict([(k, len(v)) for k, v in task_type_indices.items()])}")
        
        # Calculate MSE for each task type
        task_mse_values = {}
        valid_task_count = 0
        
        for task_type, indices in task_type_indices.items():
            if not task_type or task_type == "unknown" or len(indices) == 0:
                continue
                
            # Get pretty name for metrics
            metric_name = self.TASK_TYPE_METRIC_NAMES.get(task_type, task_type)
            
            # Calculate MSE for this task type
            total_task_mse = 0.0
            valid_samples = 0
            
            for i in indices:
                mse, success = self._calculate_mse_from_raster(filtered_responses[i], filtered_ground_truths[i])
                total_task_mse += mse
                valid_samples += 1
            
            if valid_samples > 0:
                task_mse = total_task_mse / valid_samples
                # Only include MSE per task type
                metrics[f"mse_{metric_name}"] = task_mse
                
                # Store for mean calculation
                task_mse_values[task_type] = task_mse
                valid_task_count += 1
        
        # Calculate simple mean score
        if valid_task_count > 0:
            # Simple mean (unweighted)
            simple_mean = sum(task_mse_values.values()) / valid_task_count
            metrics["primary_metric"] = simple_mean
        else:
            # Fallback to overall MSE if no task-specific metrics
            metrics["primary_metric"] = metrics["mse"]
        
        # Add compression metrics for compression tasks if needed
        if "compression" in task_type_indices and "Compression" not in [key.replace("mse_", "") for key in metrics if key.startswith("mse_")]:
            compression_indices = task_type_indices["compression"]
            compression_responses = [filtered_responses[i] for i in compression_indices]
            compression_ground_truths = [filtered_ground_truths[i] for i in compression_indices]
            
            # Only calculate the MSE for compression, not the other metrics
            compression_mse = self._calculate_compression_mse(compression_responses, compression_ground_truths)
            if compression_mse is not None:
                metrics["mse_Compression"] = compression_mse
        
        # Add num_samples for backward compatibility
        metrics["num_samples"] = len(responses)
        
        return metrics
    
    def _calculate_compression_mse(self, responses, ground_truths):
        """Calculate compression MSE only.
        
        Args:
            responses: Model-generated responses for compression tasks
            ground_truths: Ground truth answers for compression tasks
            
        Returns:
            MSE value for compression tasks, or None if no valid examples
        """
        # Count valid examples
        valid_examples = 0
        
        # Check each SVG pair
        total_mse = 0.0
        
        for response, ground_truth in zip(responses, ground_truths):
            # Clean SVGs first
            cleaned_response = self._clean_svg_text(response)
            cleaned_ground_truth = self._clean_svg_text(ground_truth)
            
            # Skip empty responses if configured
            if self.skip_empty_responses and (not cleaned_response or cleaned_response.strip() == ""):
                continue
                
            # Check rasterizability
            gt_rasterizable = self._is_rasterizable(cleaned_ground_truth)
            response_rasterizable = self._is_rasterizable(cleaned_response)
            
            # Skip or calculate MSE based on rasterizability
            if gt_rasterizable and response_rasterizable:
                # Calculate MSE between rasterized images
                mse, success = self._calculate_mse_from_raster(cleaned_response, cleaned_ground_truth)
                if success:
                    total_mse += mse
                    valid_examples += 1
        
        # Calculate metrics if we have valid examples
        if valid_examples > 0:
            return total_mse / valid_examples
        else:
            return None
            
    def _calculate_compression_metrics(self, responses, ground_truths):
        """Legacy method for compatibility, replaced by _calculate_compression_mse."""
        metrics = {}
        mse = self._calculate_compression_mse(responses, ground_truths)
        if mse is not None:
            metrics["compression_mse"] = mse
        return metrics
    
    def _setup_response_cache(self, model_name: str = "unknown", output_dir: str = "results"):
        """Set up cache for storing model responses.
        
        Args:
            model_name: Name of the model being evaluated
            output_dir: Directory to store results
        """
        from ml4s.run import get_cache_file_path
        
        # Get cache file path
        self.response_cache_file = get_cache_file_path(
            task_name="SVGEditBench",
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
        
        This is a higher-level wrapper that ensures task_types are available for metrics calculation.
        
        Args:
            model: The model to evaluate
            dataset: The dataset dictionary
            split: The split to use (e.g., "test")
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
        
        # Get input texts, targets, task types, and sample IDs
        queries, ground_truths, task_types, sample_ids = self._get_inputs_and_targets(data_split)
        
        # Generate responses
        logger.info(f"Generating responses for {len(queries)} queries")
        
        # Pass through the global_progress_bar if provided
        responses = model.generate_text(queries, sample_ids=sample_ids, **kwargs)
        
        # Save responses to the JSONL file
        for sample_id, query, response in zip(sample_ids, queries, responses):
            self._save_response_to_jsonl(sample_id, query, response, model_name)
        
        # Calculate metrics with task types included
        logger.info("Calculating metrics")
        scores = self._calculate_metrics(responses, ground_truths, queries, task_types)
        
        # Add metadata
        scores["num_samples"] = len(queries)
        
        return scores
    
    def _calculate_task_specific_metrics(self, responses, ground_truths):
        """Calculate SVG-specific metrics - not used in this implementation.
        
        The main implementation is in _calculate_metrics.
        """
        return {}
    
    def dataset_transform(self):
        """Apply any necessary transformations to the loaded dataset."""
        if not self.data_loaded:
            return
            
        # Log available columns for debugging
        for split in self.dataset:
            column_names = self.dataset[split].column_names
            logger.info(f"Dataset {split} split has columns: {column_names}")
            
            # Validate that required columns exist
            if "query" not in column_names:
                logger.warning(f"Dataset missing 'query' column in {split} split")
            if "answer" not in column_names:
                logger.warning(f"Dataset missing 'answer' column in {split} split")
            
            # Add 'id' column if not present
            if "id" not in column_names:
                logger.info(f"Adding 'id' column to dataset")
                self.dataset[split] = self.dataset[split].add_column("id", 
                                                                    [f"sample_{i}" for i in range(len(self.dataset[split]))])

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub with custom handling for original size tracking."""
        if self.data_loaded:
            return
        
        # Call the parent implementation
        super().load_data(**kwargs)
        
        # Store the original dataset sizes before any filtering
        for split in self.dataset:
            self.original_dataset_sizes[split] = len(self.dataset[split])
            logger.info(f"Original dataset size for {split} split: {self.original_dataset_sizes[split]}") 