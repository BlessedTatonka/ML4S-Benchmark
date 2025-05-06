# ML4S-Benchmark

Machine Learning for SVG Benchmark (ML4S-Benchmark) is a framework for evaluating the performance of large language models on SVG editing tasks.

## Features

- Evaluate LLMs on SVG editing tasks
- Support for both OpenRouter API and locally running Ollama models
- Standardized metrics for comparing model performance
- Caching mechanism for interrupted runs
- Comprehensive results analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ML4S-Benchmark.git
   cd ML4S-Benchmark
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Set up your OpenRouter API key (if using OpenRouter):
   ```bash
   export OPENROUTER_API_KEY=your_key_here
   ```

## Running Benchmarks

### Using OpenRouter

```bash
python -m ml4s.run \
  --model google/gemini-pro \
  --task SVGEditBench \
  --max-samples 50
```

### Using Local Ollama Models

Make sure Ollama is running on your system first. Then:

```bash
python -m ml4s.run \
  --model llama3 \
  --task SVGEditBench \
  --provider ollama \
  --max-samples 50
```

### Additional Options

- `--max-tokens`: Maximum number of tokens to generate (default: 16384)
- `--batch-size`: Number of concurrent API requests (default: 5, for Ollama always 1)
- `--output-dir`: Directory to save results (default: results)
- `--no-skip-non-rasterizable`: Don't skip non-rasterizable SVGs, instead penalize them (default: skip)
- `--no-skip-empty-responses`: Don't skip empty responses (default: skip)
- `--mock-mode`: Run without making actual API calls (for testing)
- `--ollama-url`: Specify custom URL for Ollama API (default: http://localhost:11434)

## Understanding Results

Results are stored in `results/<model_name>/<task_name>/results.json`. The metrics include:

- `total_samples`: Total number of samples evaluated
- `valid_responses`: Number of valid SVG responses
- `rasterizable_responses`: Number of SVGs that can be rendered
- `confidence`: Confidence metric based on response quality
- `mse`: Mean Squared Error (lower is better)
- `primary_metric`: Main evaluation metric
- Task-specific metrics like `mse_ChangeColor`, `mse_AddShape`, etc.

## SVG Rasterization

The benchmark uses the following standardization for comparing SVGs:
- SVGs are converted to PNG with a white background for transparent areas
- Pixel values are standardized to range [0, 1]
- MSE is calculated across all color channels and averaged, resulting in MSE values between 0 and 1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.