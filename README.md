# Solution Space Hacking

A research project focused on exploring how AI models choose which part of the solution space to explore during rollouts and training.

## Overview

This project investigates how AI models explore solution spaces through various techniques including:
- APPS dataset evaluation and analysis
- Model solution space exploration patterns
- Rollout behavior analysis
- Training data influence on solution space exploration

## Guidelines

- Do rough work in your personal scratch folder, or in `notebooks`
- Try to factor out shared utils where possible into the main repo
- Store data, weights and other large artifacts using LFS
- Use meaningful commit messages for shared stuff
- If your change is to shared stuff and possibly contentious/might affect others, use a PR and get a review
- **Package management**: Use `uv add` to add new dependencies (updates `pyproject.toml` automatically)
- **Environment variables**: Store sensitive data like API keys in `.env` files (not committed to git)

## Project Structure

```
solution-space-hacking/
├── .venv/                  # Virtual environment (created by uv)
├── config/                 # Configuration files
│   └── evaluation_config.yaml  # Evaluation pipeline settings
├── data/                   # Data storage
│   ├── apps/               # APPS dataset
│   │   ├── raw/            # Raw APPS dataset
│   │   └── cleaned/        # Cleaned and processed APPS dataset
│   ├── generation_outputs/ # Model generation outputs
│   ├── scored_outputs/     # Evaluation results
│   └── figures/            # Visualization plots
├── docs/                   # Documentation
├── logs/                   # Log files
├── models/                 # Model weights
├── notebooks/              # Notebooks with experiments
│   ├── experiments/        # Experimental notebooks
│   └── exploratory/        # Exploratory analysis notebooks
├── scratch/                # Personal scratch space
│   └── joschka/            # Personal work area
│       ├── experiments/    # Experimental code and data
│       └── notes/          # Research notes and ideas
├── src/                    # Source code
│   ├── apps/               # APPS dataset handling
│   ├── evaluation/         # Model evaluation utilities
│   ├── openrouter/         # OpenRouter API integration
│   ├── utils/              # Utility functions
│   └── visualization/      # Results visualization
├── tests/                  # Test files
├── run_generation.py       # Model generation script
├── run_evaluation.py       # Evaluation script
├── run_full_pipeline.py    # Full pipeline script
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Locked dependency versions
└── .env                    # Environment variables (not in git)
```

## Key Modules

### APPS Module (`src/apps/`)
- **Dataset Loading**: Load and preprocess APPS dataset
- **Problem Generation**: Generate coding problems from APPS
- **Solution Analysis**: Analyze model solutions and exploration patterns

### Evaluation Module (`src/evaluation/`)
- **Model Evaluation**: Evaluate models on APPS coding problems
- **Solution Space Analysis**: Analyze how models explore solution spaces
- **Rollout Analysis**: Analyze model behavior during rollouts

### OpenRouter Module (`src/openrouter/`)
- **API Integration**: Integrate with OpenRouter API for model inference
- **Model Selection**: Interface with various models available on OpenRouter
- **Response Processing**: Process and analyze model responses

## Package Management

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

### Setup

1. **Install uv** (if not already installed):
   ```bash
   brew install uv  # macOS
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run scripts**:
   ```bash
   uv run python your_script.py
   ```

### Key Commands

```bash
# Add a new dependency
uv add package_name

# Add a dependency with version constraint
uv add "package_name>=1.0.0"

# Install all dependencies
uv sync

# Run a script in the virtual environment
uv run python script.py

# Activate virtual environment manually
source .venv/bin/activate

# Deactivate virtual environment
deactivate
```

### Project Configuration

- **Python version**: `>=3.11` (see `pyproject.toml`)
- **Virtual environment**: `.venv/` (created by uv)
- **Dependencies**: Listed in `pyproject.toml` and locked in `uv.lock`

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd solution-space-hacking
   ```

2. **Set up the environment**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env  # if available
   # Edit .env with your API keys
   ```

4. **Run experiments**:
   ```bash
   # Generate model outputs
   uv run python run_generation.py --n-problems 50 --split eval
   
   # Evaluate existing outputs
   uv run python run_evaluation.py --input-file data/generation_outputs/latest_file.json
   
   # Run full pipeline (generation + evaluation)
   uv run python run_full_pipeline.py --n-problems 50 --split eval
   
   # Personal experiments
   uv run python scratch/joschka/experiments/your_experiment.py
   ```

## Data Organization

- **APPS Dataset**: Raw APPS dataset stored in `data/apps/raw/`, cleaned version in `data/apps/cleaned/`
- **Generation Outputs**: Model generation results stored in `data/generation_outputs/`
- **Evaluation Results**: Scored outputs and metrics in `data/scored_outputs/`
- **Visualizations**: Plots and figures in `data/figures/`
- **Experiments**: Experimental datasets in `data/experiments/`

## Documentation

- **General**: See `docs/` for project documentation
- **Project Status**: Current implementation status in `docs/PROJECT_STATUS.md`
- **APPS**: APPS dataset documentation in `docs/apps/`
- **Evaluation**: Evaluation methodology in `docs/evaluation/`

## Contributing

1. Create a personal scratch folder in `scratch/your_name/`
2. Use meaningful commit messages
3. For shared code changes, create a pull request
4. Follow the project guidelines for code organization

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Joschka Braun - [GitHub Profile](https://github.com/JoschkaCBraun) 