#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import asyncio
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from functools import partialmethod
import time
import requests

from openai import AsyncOpenAI, OpenAI
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Setup logging - make it less verbose by default
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress httpx logging (to avoid printing HTTP request details)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# Suppress tqdm logging of progress bars
logging.getLogger("tqdm").setLevel(logging.WARNING)

# Task mapping
TASKS = {
    "SVGEditBench": "ml4s.tasks.Editing.SVGEditBench",
    "SGPBench": "ml4s.tasks.Understanding.SGPBench",
}

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

def signal_handler(sig, frame):
    """Handle interrupt signals for graceful shutdown."""
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        logger.warning("Forced shutdown requested. Exiting immediately.")
        sys.exit(1)
    
    logger.warning("Interrupt received. Finishing current batch and shutting down...")
    SHUTDOWN_REQUESTED = True


class OpenRouterModelInterface:
    """Interface for OpenRouter models."""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        response_cache_file: Optional[str] = None,
        batch_size: int = 5,
        mock_mode: bool = False,
        skip_empty_responses: bool = False,
        **kwargs,
    ):
        """Initialize the OpenRouter model interface.
        
        Args:
            model_name: The model name on OpenRouter
            api_key: OpenRouter API key (defaults to env var OPENROUTER_API_KEY)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature is forced to 0.0 for deterministic outputs
            response_cache_file: Path to file storing cached responses
            batch_size: Number of concurrent requests to make
            mock_mode: If True, no API calls will be made, mock responses returned
            skip_empty_responses: If True, don't retry empty responses
            **kwargs: Additional parameters for the model
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.mock_mode = mock_mode
        self.skip_empty_responses = skip_empty_responses
        
        if not self.api_key and not self.mock_mode:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter. Alternatively, use --mock-mode for testing without an API key."
            )
            
        self.max_tokens = max_tokens
        # Force temperature to 0.0 for deterministic outputs
        self.temperature = 0.0
        self.model_kwargs = kwargs
        self.batch_size = batch_size
        
        # Cached responses
        self.response_cache_file = response_cache_file
        self.cached_responses = {}
        
        # Load cached responses if file exists
        if response_cache_file and os.path.exists(response_cache_file):
            self._load_cached_responses()
        
        # Configure OpenAI clients for OpenRouter (skip if mock mode)
        if not self.mock_mode:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            
            self.async_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
        
        # Set HTTP headers
        self.http_headers = {
            "HTTP-Referer": "https://github.com/your-username/ML4S-Benchmark",
            "X-Title": "ML4S-Benchmark",
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _load_cached_responses(self):
        """Load cached responses from JSONL file."""
        self.cached_responses = {}
        try:
            with open(self.response_cache_file, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # Use sample_id as key
                        if 'sample_id' in item and 'response' in item:
                            self.cached_responses[item['sample_id']] = item
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line in cache file: {line}")
            
            logger.info(f"Loaded {len(self.cached_responses)} cached responses from {self.response_cache_file}")
        except Exception as e:
            logger.warning(f"Error loading cached responses: {e}")
    
    def _save_response_to_cache(self, sample_id: str, query: str, response: str):
        """Save a response to the cache file."""
        if not self.response_cache_file:
            return
        
        # Get existing retry count if any
        retry_count = 0
        if sample_id in self.cached_responses:
            retry_count = self.cached_responses.get(sample_id, {}).get('retry_count', 0)
        
        # Create cache entry
        cache_entry = {
            'sample_id': sample_id,
            'query': query,
            'response': response,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'retry_count': retry_count
        }
        
        # Save to cache dictionary
        self.cached_responses[sample_id] = cache_entry
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.response_cache_file)), exist_ok=True)
        
        # Append to file - each response is saved individually
        with open(self.response_cache_file, 'a') as f:
            f.write(json.dumps(cache_entry) + '\n')
            f.flush()  # Ensure it's written to disk immediately
    
    async def _generate_completion(self, prompt: str, sample_id: str):
        """Generate a single completion asynchronously."""
        if self.mock_mode:
            # Return a mock response
            text = f"<svg>Mock SVG response for {sample_id}</svg>"
            # Save to cache immediately
            self._save_response_to_cache(sample_id, prompt, text)
            return sample_id, prompt, text, None
            
        try:
            # Add timeout to avoid hanging requests
            response = await asyncio.wait_for(
                self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_headers=self.http_headers,
                    **self.model_kwargs,
                ),
                timeout=60  # 60 second timeout
            )
            
            # Check if response is None or missing expected attributes
            if response is None:
                logger.error(f"Received None response for sample {sample_id}")
                # Save empty response to cache
                self._save_response_to_cache(sample_id, prompt, "")
                return sample_id, prompt, "", "Received None response"
                
            # Check if response has the expected structure
            if not hasattr(response, 'choices') or not response.choices:
                logger.error(f"Response missing choices array for sample {sample_id}")
                # Save empty response to cache
                self._save_response_to_cache(sample_id, prompt, "")
                return sample_id, prompt, "", "Response missing choices"
                
            # Check if the first choice has message and content
            if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                logger.error(f"Response has invalid structure for sample {sample_id}")
                # Save empty response to cache
                self._save_response_to_cache(sample_id, prompt, "")
                return sample_id, prompt, "", "Response has invalid structure"
            
            # Extract the content
            text = response.choices[0].message.content
            
            # Handle None or empty text - don't log a warning, just save empty text
            if text is None or text.strip() == "":
                text = ""
            
            # Save to cache immediately after getting the response
            self._save_response_to_cache(sample_id, prompt, text)
            
            return sample_id, prompt, text, None
            
        except asyncio.TimeoutError:
            logger.error(f"Request timed out for sample {sample_id}")
            # Save empty response to cache to prevent retrying endlessly
            self._save_response_to_cache(sample_id, prompt, "")
            return sample_id, prompt, "", "Request timed out"
            
        except Exception as e:
            logger.error(f"Error generating response for sample {sample_id}: {e}")
            # Save empty response to cache to prevent retrying endlessly
            self._save_response_to_cache(sample_id, prompt, "")
            return sample_id, prompt, "", str(e)
    
    async def _generate_completions_batch(self, prompts, sample_ids):
        """Generate multiple completions in parallel.
        
        Each completion is saved to cache immediately after it completes.
        """
        tasks = [self._generate_completion(prompt, sample_id) 
                for prompt, sample_id in zip(prompts, sample_ids)]
        
        # As each task completes, it's already saved to cache
        results = await asyncio.gather(*tasks)
        return results
    
    def generate_text(self, prompts: List[str], sample_ids: List[str] = None, max_retries: int = 1, show_progress: bool = False, **kwargs) -> List[str]:
        """Generate text from the model for the given prompts.
        
        Args:
            prompts: List of prompts
            sample_ids: List of sample identifiers for caching
            max_retries: Maximum number of retries for failed requests (default: 1)
            show_progress: Whether to show progress bars for batches
            **kwargs: Additional parameters
            
        Returns:
            List of generated texts
        """
        # Default sample_ids if not provided
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(prompts))]
        
        responses = [None] * len(prompts)  # Initialize with None placeholders
        to_generate = []
        to_generate_indices = []
        to_generate_sample_ids = []
        
        # Check which responses we already have in cache
        for i, (prompt, sample_id) in enumerate(zip(prompts, sample_ids)):
            if sample_id in self.cached_responses:
                cached_response = self.cached_responses[sample_id]
                if cached_response.get('retry_count', 0) < max_retries:
                    # Use cached response if it exists and hasn't exceeded max retries
                    logger.debug(f"Using cached response for sample {sample_id}")
                    
                    # If the response is empty and retry_count < max_retries, retry it
                    # Skip retrying if skip_empty_responses is True
                    if (not cached_response['response'] or cached_response['response'].strip() == "") and not self.skip_empty_responses:
                        logger.debug(f"Empty response for {sample_id}, retrying (attempt {cached_response.get('retry_count', 0) + 1}/{max_retries})")
                        to_generate.append(prompt)
                        to_generate_indices.append(i)
                        to_generate_sample_ids.append(sample_id)
                    else:
                        responses[i] = cached_response['response']
                else:
                    # If we've already retried max times, use the cached response even if empty
                    logger.debug(f"Using cached response for sample {sample_id} after max retries")
                    responses[i] = cached_response['response']
            else:
                # Need to generate this response
                to_generate.append(prompt)
                to_generate_indices.append(i)
                to_generate_sample_ids.append(sample_id)
        
        # Get global progress bar if it exists
        progress_bar = kwargs.pop('global_progress_bar', None)
        
        # Generate new responses if needed
        if to_generate:
            logger.info(f"Generating {len(to_generate)} new responses in {(len(to_generate) + self.batch_size - 1) // self.batch_size} batches")
            
            # Process in batches using asyncio
            loop = asyncio.get_event_loop()
            
            # Create a mapping of sample_id to original index for faster lookup
            sample_id_to_idx = {sample_id: idx for idx, sample_id in enumerate(to_generate_sample_ids)}
            
            # Process in smaller batches to avoid overwhelming the API
            for i in range(0, len(to_generate), self.batch_size):
                # Check if shutdown was requested
                global SHUTDOWN_REQUESTED
                if SHUTDOWN_REQUESTED:
                    logger.warning("Shutdown requested. Stopping further API calls.")
                    break
                
                batch_prompts = to_generate[i:i+self.batch_size]
                batch_ids = to_generate_sample_ids[i:i+self.batch_size]
                
                batch_size = len(batch_prompts)
                total_batches = (len(to_generate) + self.batch_size - 1) // self.batch_size
                current_batch = i // self.batch_size + 1
                
                # Only log detailed batch info in debug level
                logger.debug(f"Processing batch {current_batch}/{total_batches} ({batch_size} items)")
                
                # Process batch with direct async calls
                tasks = [self._generate_completion(prompt, sample_id) 
                        for prompt, sample_id in zip(batch_prompts, batch_ids)]
                
                batch_results = loop.run_until_complete(asyncio.gather(*tasks))
                
                # Process results and update progress
                for sample_id, prompt, text, error in batch_results:
                    # Get the index using the mapping instead of list.index() which is O(n)
                    batch_idx = sample_id_to_idx.get(sample_id)
                    if batch_idx is not None:
                        gen_idx = to_generate_indices[batch_idx]
                        
                        # Update responses list (the response is already cached)
                        responses[gen_idx] = text
                        
                        # Update retry count in cache
                        if error is not None:
                            retry_count = self.cached_responses.get(sample_id, {}).get('retry_count', 0)
                            self.cached_responses[sample_id]['retry_count'] = retry_count + 1
                    else:
                        logger.warning(f"Could not find index for sample_id {sample_id} in results processing")
                
                # Update the global progress bar if it exists
                if progress_bar is not None:
                    progress_bar.update(batch_size)
                    # Force refresh the progress bar
                    progress_bar.refresh()
                    # Small delay to ensure the progress bar is displayed
                    time.sleep(0.01)
        else:
            logger.info("All responses already cached, no new generations needed")
            
            # Update progress bar for cached responses if it exists
            if progress_bar is not None:
                progress_bar.update(len(prompts))
                progress_bar.refresh()
        
        # Replace any None values with empty strings (for samples that weren't processed due to shutdown)
        responses = [r if r is not None else "" for r in responses]
                
        return responses


class OllamaModelInterface:
    """Interface for models running locally with Ollama."""
    
    def __init__(
        self,
        model_name: str,
        api_url: str = "http://localhost:11434",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        response_cache_file: Optional[str] = None,
        batch_size: int = 1,  # Default to 1 for Ollama which doesn't support batch requests
        mock_mode: bool = False,
        skip_empty_responses: bool = False,
        **kwargs,
    ):
        """Initialize the Ollama model interface.
        
        Args:
            model_name: The model name in Ollama
            api_url: Ollama API URL (defaults to http://localhost:11434)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature is forced to 0.0 for deterministic outputs
            response_cache_file: Path to file storing cached responses
            batch_size: Number of concurrent requests (note: Ollama doesn't support batch requests)
            mock_mode: If True, no API calls will be made, mock responses returned
            skip_empty_responses: If True, don't retry empty responses
            **kwargs: Additional parameters for the model
        """
        self.model_name = model_name
        self.api_url = api_url
        self.mock_mode = mock_mode
        self.skip_empty_responses = skip_empty_responses
        
        # Force batch size to 1 for Ollama (doesn't support batch requests)
        self.batch_size = 1
        logger.info(f"Note: Ollama doesn't support batch requests, using batch_size=1")
        
        self.max_tokens = max_tokens
        # Force temperature to 0.0 for deterministic outputs
        self.temperature = 0.0
        self.model_kwargs = kwargs
        
        # Cached responses
        self.response_cache_file = response_cache_file
        self.cached_responses = {}
        
        # Load cached responses if file exists
        if response_cache_file and os.path.exists(response_cache_file):
            self._load_cached_responses()
        
        # Check if the model is available in Ollama
        if not self.mock_mode:
            try:
                response = requests.get(f"{self.api_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name") for model in models]
                    if self.model_name not in model_names:
                        logger.warning(f"Model '{self.model_name}' not found in Ollama. Available models: {model_names}")
                        logger.warning(f"You may need to pull the model first with: ollama pull {self.model_name}")
                else:
                    logger.warning(f"Failed to get available models from Ollama: {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to connect to Ollama: {e}")
                logger.warning("Make sure Ollama is running on your system.")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _load_cached_responses(self):
        """Load cached responses from JSONL file."""
        self.cached_responses = {}
        try:
            with open(self.response_cache_file, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # Use sample_id as key
                        if 'sample_id' in item and 'response' in item:
                            self.cached_responses[item['sample_id']] = item
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line in cache file: {line}")
            
            logger.info(f"Loaded {len(self.cached_responses)} cached responses from {self.response_cache_file}")
        except Exception as e:
            logger.warning(f"Error loading cached responses: {e}")
    
    def _save_response_to_cache(self, sample_id: str, query: str, response: str):
        """Save a response to the cache file."""
        if not self.response_cache_file:
            return
        
        # Get existing retry count if any
        retry_count = 0
        if sample_id in self.cached_responses:
            retry_count = self.cached_responses.get(sample_id, {}).get('retry_count', 0)
        
        # Create cache entry
        cache_entry = {
            'sample_id': sample_id,
            'query': query,
            'response': response,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'retry_count': retry_count
        }
        
        # Save to cache dictionary
        self.cached_responses[sample_id] = cache_entry
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.response_cache_file)), exist_ok=True)
        
        # Append to file - each response is saved individually
        with open(self.response_cache_file, 'a') as f:
            f.write(json.dumps(cache_entry) + '\n')
            f.flush()  # Ensure it's written to disk immediately
    
    async def _generate_completion(self, prompt: str, sample_id: str):
        """Generate a single completion."""
        if self.mock_mode:
            # Return a mock response
            text = f"<svg>Mock SVG response for {sample_id}</svg>"
            # Save to cache immediately
            self._save_response_to_cache(sample_id, prompt, text)
            return sample_id, prompt, text, None
            
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            # Make API request to Ollama
            response = requests.post(f"{self.api_url}/api/generate", json=payload)
            
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                # Save empty response to cache
                self._save_response_to_cache(sample_id, prompt, "")
                return sample_id, prompt, "", error_msg
            
            # Extract the response text
            response_json = response.json()
            text = response_json.get("response", "")
            
            # Handle None or empty text
            if text is None or text.strip() == "":
                text = ""
            
            # Save to cache immediately
            self._save_response_to_cache(sample_id, prompt, text)
            
            return sample_id, prompt, text, None
            
        except Exception as e:
            logger.error(f"Error generating response for sample {sample_id}: {e}")
            # Save empty response to cache to prevent retrying endlessly
            self._save_response_to_cache(sample_id, prompt, "")
            return sample_id, prompt, "", str(e)
    
    async def _generate_completions_batch(self, prompts, sample_ids):
        """Generate multiple completions (Ollama only supports one at a time)."""
        tasks = [self._generate_completion(prompt, sample_id) 
                for prompt, sample_id in zip(prompts, sample_ids)]
        
        # Run the tasks one by one (Ollama doesn't support concurrent requests well)
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        
        return results
    
    def generate_text(self, prompts: List[str], sample_ids: List[str] = None, max_retries: int = 1, show_progress: bool = False, **kwargs) -> List[str]:
        """Generate text from the model for the given prompts.
        
        Args:
            prompts: List of prompts
            sample_ids: List of sample identifiers for caching
            max_retries: Maximum number of retries for failed requests (default: 1)
            show_progress: Whether to show progress bars for batches
            **kwargs: Additional parameters
            
        Returns:
            List of generated texts
        """
        # Default sample_ids if not provided
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(prompts))]
        
        responses = [None] * len(prompts)  # Initialize with None placeholders
        to_generate = []
        to_generate_indices = []
        to_generate_sample_ids = []
        
        # Check which responses we already have in cache
        for i, (prompt, sample_id) in enumerate(zip(prompts, sample_ids)):
            if sample_id in self.cached_responses:
                cached_response = self.cached_responses[sample_id]
                if cached_response.get('retry_count', 0) < max_retries:
                    # Use cached response if it exists and hasn't exceeded max retries
                    logger.debug(f"Using cached response for sample {sample_id}")
                    
                    # If the response is empty and retry_count < max_retries, retry it
                    # Skip retrying if skip_empty_responses is True
                    if (not cached_response['response'] or cached_response['response'].strip() == "") and not self.skip_empty_responses:
                        logger.debug(f"Empty response for {sample_id}, retrying (attempt {cached_response.get('retry_count', 0) + 1}/{max_retries})")
                        to_generate.append(prompt)
                        to_generate_indices.append(i)
                        to_generate_sample_ids.append(sample_id)
                    else:
                        responses[i] = cached_response['response']
                else:
                    # If we've already retried max times, use the cached response even if empty
                    logger.debug(f"Using cached response for sample {sample_id} after max retries")
                    responses[i] = cached_response['response']
            else:
                # Need to generate this response
                to_generate.append(prompt)
                to_generate_indices.append(i)
                to_generate_sample_ids.append(sample_id)
        
        # Get global progress bar if it exists
        progress_bar = kwargs.pop('global_progress_bar', None)
        
        # Generate new responses if needed
        if to_generate:
            logger.info(f"Generating {len(to_generate)} new responses using Ollama model {self.model_name}")
            
            # Process in single requests (Ollama doesn't support batching)
            loop = asyncio.get_event_loop()
            
            # Create a mapping of sample_id to original index for faster lookup
            sample_id_to_idx = {sample_id: idx for idx, sample_id in enumerate(to_generate_sample_ids)}
            
            # Process each prompt individually
            for i in range(0, len(to_generate)):
                # Check if shutdown was requested
                global SHUTDOWN_REQUESTED
                if SHUTDOWN_REQUESTED:
                    logger.warning("Shutdown requested. Stopping further API calls.")
                    break
                
                prompt = to_generate[i]
                sample_id = to_generate_sample_ids[i]
                
                current_item = i + 1
                total_items = len(to_generate)
                
                # Only log the progress to avoid spamming
                logger.debug(f"Processing item {current_item}/{total_items}")
                
                # Process a single prompt
                task = self._generate_completion(prompt, sample_id)
                sample_id, prompt, text, error = loop.run_until_complete(task)
                
                # Get the index using the mapping
                batch_idx = sample_id_to_idx.get(sample_id)
                if batch_idx is not None:
                    gen_idx = to_generate_indices[batch_idx]
                    
                    # Update responses list (the response is already cached)
                    responses[gen_idx] = text
                    
                    # Update retry count in cache
                    if error is not None:
                        retry_count = self.cached_responses.get(sample_id, {}).get('retry_count', 0)
                        self.cached_responses[sample_id]['retry_count'] = retry_count + 1
                else:
                    logger.warning(f"Could not find index for sample_id {sample_id} in results processing")
                
                # Update the global progress bar if it exists
                if progress_bar is not None:
                    progress_bar.update(1)
                    # Force refresh the progress bar
                    progress_bar.refresh()
                    # Small delay to ensure the progress bar is displayed
                    time.sleep(0.01)
        else:
            logger.info("All responses already cached, no new generations needed")
            
            # Update progress bar for cached responses if it exists
            if progress_bar is not None:
                progress_bar.update(len(prompts))
                progress_bar.refresh()
        
        # Replace any None values with empty strings (for samples that weren't processed due to shutdown)
        responses = [r if r is not None else "" for r in responses]
                
        return responses


def load_task(task_name: str) -> Any:
    """Load a task by name.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Task instance
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASKS.keys())}")
    
    module_path = TASKS[task_name]
    module_name, class_name = module_path.rsplit(".", 1)
    
    try:
        module = importlib.import_module(module_name)
        task_class = getattr(module, class_name)
        return task_class()
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load task {task_name}: {e}")


def save_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """Save evaluation results to a file.
    
    Args:
        results: Evaluation results
        output_dir: Directory to save results
        
    Returns:
        Path to the saved results file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract task and model information
    task_name = results["metadata"]["task_name"]
    model_name = results["metadata"]["model_name"]
    
    # Create model-specific directory structure (new format: results/<model_name>/<task_name>/)
    model_path = Path(model_name)
    
    # Handle the case where model_name might have multiple parts (provider/model)
    if len(model_path.parts) > 1:
        # If model name is already in format provider/model, use it as is
        model_folder = Path(output_dir) / model_path / task_name
    else:
        # If it's just a model name without provider, use it directly
        model_folder = Path(output_dir) / model_name / task_name
    
    # Create the directories
    os.makedirs(model_folder, exist_ok=True)
    
    # Create output path for the results file
    output_path = model_folder / "results.json"
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Results saved to {output_path}")
    return str(output_path)


def get_cache_file_path(task_name: str, model_name: str, output_dir: str = "results") -> str:
    """Get the path to the cache file for a task and model.
    
    Args:
        task_name: Name of the task
        model_name: Name of the model
        output_dir: Output directory
        
    Returns:
        Path to the cache file
    """
    # Create directory structure in the new format results/<model_name>/<task_name>/
    model_path = Path(model_name)
    
    # Handle the case where model_name might have multiple parts (provider/model)
    if len(model_path.parts) > 1:
        # If model name is already in format provider/model, use it as is
        model_folder = Path(output_dir) / model_path / task_name
    else:
        # If it's just a model name without provider, use it directly
        model_folder = Path(output_dir) / model_name / task_name
    
    # Create the directories
    os.makedirs(model_folder, exist_ok=True)
    
    # Create file path
    return str(model_folder / "responses.jsonl")


def process_dataset_with_model(model, dataset, split, task, **kwargs):
    """Process a dataset with a model and return the results.
    
    This function handles extracting data, generating responses, and calculating metrics.
    
    Args:
        model: The model interface
        dataset: The dataset to process
        split: The dataset split to use
        task: The task object
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with results
    """
    # If the task has a specialized process_dataset_with_model method, use it
    if hasattr(task, 'process_dataset_with_model'):
        logger.info(f"Using task's specialized process_dataset_with_model method")
        
        # We need to handle progress bars even for specialized tasks
        data_split = dataset[split]
        total_items = len(data_split)
        
        # Create a progress bar that will be visible throughout processing
        show_progress = not kwargs.get('no_progress', False)
        progress_bar = tqdm(total=total_items, desc="Evaluating samples", disable=not show_progress)
        
        try:
            # Add the progress bar to kwargs
            kwargs['global_progress_bar'] = progress_bar
            # Call the task's method
            result = task.process_dataset_with_model(model, dataset, split, **kwargs)
            return result
        finally:
            # Make sure to close the progress bar
            progress_bar.close()
    
    # Otherwise, use the default implementation
    # Get dataset split
    data_split = dataset[split]
    
    # For SGPBench, call the task's evaluate method directly (requires 3 values)
    if task.__class__.__name__ == "SGPBench":
        logger.info(f"Using task-specific evaluation for SGPBench")
        return task.evaluate(model, split, **kwargs)
    
    # Get input texts and targets
    queries, ground_truths = task._get_inputs_and_targets(data_split)
    
    # Get sample IDs
    sample_ids = None
    if "id" in data_split.column_names:
        sample_ids = data_split["id"]
        logger.info(f"Using {len(sample_ids)} sample IDs from dataset")
    else:
        sample_ids = [f"{split}_sample_{i}" for i in range(len(queries))]
        logger.info(f"Generated {len(sample_ids)} sample IDs")
    
    # Generate responses
    logger.info(f"Generating responses for {len(queries)} queries")
    
    # Set retries to 1 by default
    kwargs['max_retries'] = kwargs.get('max_retries', 1)
    
    # Create a visible progress bar
    show_progress = not kwargs.get('no_progress', False)
    with tqdm(total=len(queries), desc="Evaluating samples", disable=not show_progress) as progress_bar:
        # Add the progress bar to kwargs to be used by generate_text
        kwargs['global_progress_bar'] = progress_bar
        
        # Generate responses
        responses = model.generate_text(queries, sample_ids=sample_ids, **kwargs)
    
    # Calculate metrics
    logger.info("Calculating metrics")
    scores = task._calculate_metrics(responses, ground_truths)
    
    # Add metadata
    scores["num_samples"] = len(queries)
    
    return scores


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the evaluation.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Evaluation results
    """
    # Add no_progress to kwargs based on args.no_progress
    kwargs = {'no_progress': args.no_progress}
    
    # Load the task
    task = load_task(args.task)
    
    # Set task-specific options
    if args.task == "SVGEditBench":
        # By default, we skip both non-rasterizable SVGs and empty responses (don't penalize)
        # The flags can be used to disable this behavior
        task.penalize_non_rasterizable = args.no_skip_non_rasterizable
        task.skip_empty_responses = not args.no_skip_empty_responses
    
    # Get cache file path
    cache_file = get_cache_file_path(args.task, args.model, args.output_dir)
    
    # Create model interface with cache file based on provider
    if args.provider == "ollama":
        # Create Ollama model interface
        model = OllamaModelInterface(
            model_name=args.model,
            api_url=args.ollama_url,
            max_tokens=args.max_tokens,
            temperature=0.0,  # Force temperature to 0
            response_cache_file=cache_file,
            batch_size=1,  # Ollama doesn't support batch requests
            mock_mode=args.mock_mode,
            skip_empty_responses=not args.no_skip_empty_responses if args.task == "SVGEditBench" else False,
        )
    else:
        # Default to OpenRouter model interface
        model = OpenRouterModelInterface(
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=0.0,  # Force temperature to 0
            response_cache_file=cache_file,
            batch_size=args.batch_size,
            mock_mode=args.mock_mode,
            skip_empty_responses=not args.no_skip_empty_responses if args.task == "SVGEditBench" else False,
        )
    
    # Load dataset
    task.load_data()
    
    # Always shuffle the dataset with seed 42 for consistency
    logger.info("Shuffling dataset with seed 42")
    import random
    random_gen = random.Random(42)  # Use fixed seed 42
    
    for split in task.dataset:
        # Get the total number of samples
        total_samples = len(task.dataset[split])
        
        # Shuffle the indices
        shuffled_indices = list(range(total_samples))
        random_gen.shuffle(shuffled_indices)
        
        # Apply the shuffled indices to the dataset
        task.dataset[split] = task.dataset[split].select(shuffled_indices)
        logger.info(f"Shuffled {total_samples} samples in {split} split")
    
    # Handle sample limiting by selecting max_samples
    if args.max_samples is not None:
        logger.info(f"Selecting first {args.max_samples} samples from shuffled dataset")
        for split in task.dataset:
            # Get the total number of samples
            total_samples = len(task.dataset[split])
            # Cap the sample count at the dataset size
            n = min(args.max_samples, total_samples)
            
            # Select the first n samples from the already shuffled dataset
            task.dataset[split] = task.dataset[split].select(range(n))
            logger.info(f"Selected {len(task.dataset[split])} samples from {split} split (from total {total_samples})")
    
    # Limit samples for test mode (keep this for backward compatibility)
    elif args.test_mode:
        logger.info(f"Running in test mode with {args.test_samples} samples")
        for split in task.dataset:
            # Slice dataset to first n samples (n = test_samples)
            n = min(args.test_samples, len(task.dataset[split]))
            task.dataset[split] = task.dataset[split].select(range(n))
    
    # Run evaluation
    logger.info(f"Running evaluation for task '{args.task}' with model '{args.model}'")
    
    # Use task-specific default split if available
    split = args.split
    if args.task == "SGPBench" and split == "test":
        # Use the first available split for SGPBench
        split = list(task.dataset.keys())[0]
        logger.info(f"Using '{split}' split for SGPBench task instead of 'test'")
    
    scores = process_dataset_with_model(model, task.dataset, split, task, **kwargs)
    
    # Create metadata
    metadata = {
        "task_name": args.task,
        "model_name": args.model,
        "split": args.split,
        "timestamp": datetime.now().isoformat(),
        "model_params": {
            "max_tokens": args.max_tokens,
            "temperature": 0.0,  # Force temperature to 0
            "batch_size": args.batch_size,
        },
        "task_params": {
            "penalize_non_rasterizable": args.no_skip_non_rasterizable if args.task == "SVGEditBench" else None,
            "skip_empty_responses": not args.no_skip_empty_responses if args.task == "SVGEditBench" else None,
        },
        "test_mode": args.test_mode,
        "test_samples": args.test_samples if args.test_mode else None,
        "max_samples": args.max_samples,
        "response_cache_file": cache_file,
        "mock_mode": args.mock_mode,
    }
    
    # Create final results
    results = {
        "metadata": metadata,
        "scores": scores,
    }
    
    # Save results if requested
    if not args.no_save:
        save_results(results, args.output_dir)
        
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "ML4S-Benchmark: Run evaluations for SVG-related tasks\n\n"
            "INTERRUPT HANDLING: You can safely interrupt with Ctrl+C at any time.\n"
            "The current batch will finish processing, and all completed responses\n"
            "will be saved. You can resume later by running the same command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model name (either on OpenRouter or locally via Ollama)",
    )
    parser.add_argument(
        "-t", "--task",
        required=True,
        choices=list(TASKS.keys()),
        help="Task to evaluate",
    )
    
    # Provider options
    parser.add_argument(
        "--provider",
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="Model provider to use (default: openrouter)",
    )
    
    # Ollama-specific options
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )
    
    # Optional arguments
    parser.add_argument(
        "-s", "--split",
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens to generate (default: 16384)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )
    
    # Sample selection arguments
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate, randomly selected with seed 42",
    )
    
    # Test mode arguments (kept for backward compatibility)
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with a limited number of samples",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=5,
        help="Number of samples to run in test mode (default: 5)",
    )
    
    # Mock mode argument
    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Run in mock mode without making API calls",
    )
    
    # SVGEditBench specific arguments
    parser.add_argument(
        "--no-skip-non-rasterizable",
        action="store_true",
        help="Don't skip non-rasterizable SVGs, instead penalize them with high MSE (by default they're skipped)",
    )
    parser.add_argument(
        "--no-skip-empty-responses",
        action="store_true",
        help="Don't skip empty responses in metrics calculation (by default they're skipped)",
    )
    
    # Add the show_progress argument to the parser
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars for batch processing",
    )
    
    # Add the no_progress argument to the parser
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide all progress bars (both global and batch-level)",
    )
    
    args = parser.parse_args()
    
    try:
        results = run_evaluation(args)
        if args.no_save:
            print(json.dumps(results, indent=2))
        
        # Print summary
        scores = results["scores"]
        main_metric = scores.get("mse", None)
        if main_metric is not None:
            logger.info(f"Evaluation completed. MSE: {main_metric:.6f}, Samples: {scores.get('num_samples', 0)}")
        else:
            logger.info(f"Evaluation completed. Samples: {scores.get('num_samples', 0)}")
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user. Partial results have been saved.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 