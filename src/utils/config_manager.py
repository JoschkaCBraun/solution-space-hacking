"""
Centralized configuration management for the evaluation pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    split: str = "eval"
    n_problems: int = 10
    difficulty: str = "introductory"
    sample_types: List[str] = field(default_factory=lambda: [""])


@dataclass
class ModelConfig:
    """Model configuration."""
    models: List[str] = field(default_factory=list)
    max_workers: int = 10
    timeout: int = 180
    max_tokens: int = 4096
    temperature: float = 0.1


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "data"
    generation_dir: str = "generation_outputs"
    scored_dir: str = "scored_outputs"
    figures_dir: str = "figures"


class ConfigManager:
    """Manages configuration for the evaluation pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        self.config_path = config_path or "config/evaluation_config.yaml"
        self.dataset_config = DatasetConfig()
        self.model_config = ModelConfig()
        self.output_config = OutputConfig()
        
        if config_path and Path(config_path).exists():
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load dataset config
        if 'dataset' in config:
            dataset = config['dataset']
            self.dataset_config.split = dataset.get('split', self.dataset_config.split)
            self.dataset_config.n_problems = dataset.get('n_problems', self.dataset_config.n_problems)
            self.dataset_config.difficulty = dataset.get('difficulty', self.dataset_config.difficulty)
            self.dataset_config.sample_types = dataset.get('sample_types', self.dataset_config.sample_types)
        
        # Load model config
        if 'models' in config:
            models = config['models']
            self.model_config.models = models.get('models', self.model_config.models)
            self.model_config.max_workers = models.get('max_workers', self.model_config.max_workers)
            
        if 'generation' in config:
            generation = config['generation']
            self.model_config.max_tokens = generation.get('max_tokens', self.model_config.max_tokens)
            self.model_config.temperature = generation.get('temperature', self.model_config.temperature)
            self.model_config.timeout = generation.get('timeout', self.model_config.timeout)
        
        # Load output config
        if 'output' in config:
            output = config['output']
            self.output_config.base_dir = output.get('base_dir', self.output_config.base_dir)
            self.output_config.generation_dir = output.get('generation_dir', self.output_config.generation_dir)
            self.output_config.scored_dir = output.get('scored_dir', self.output_config.scored_dir)
            self.output_config.figures_dir = output.get('figures_dir', self.output_config.figures_dir)
    
    def update_from_args(self, args: Any) -> None:
        """Update configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Update dataset config
        if hasattr(args, 'split') and args.split:
            self.dataset_config.split = args.split
        if hasattr(args, 'n_problems') and args.n_problems:
            self.dataset_config.n_problems = args.n_problems
        if hasattr(args, 'difficulty') and args.difficulty:
            self.dataset_config.difficulty = args.difficulty
            
        # Update model config
        if hasattr(args, 'models') and args.models:
            self.model_config.models = args.models
        if hasattr(args, 'max_workers') and args.max_workers:
            self.model_config.max_workers = args.max_workers
        if hasattr(args, 'timeout') and args.timeout:
            self.model_config.timeout = args.timeout
        if hasattr(args, 'max_tokens') and args.max_tokens:
            self.model_config.max_tokens = args.max_tokens
        if hasattr(args, 'temperature') and args.temperature:
            self.model_config.temperature = args.temperature
    
    def get_generation_path(self) -> Path:
        """Get the path for generation outputs."""
        return Path(self.output_config.base_dir) / self.output_config.generation_dir
    
    def get_scored_path(self) -> Path:
        """Get the path for scored outputs."""
        return Path(self.output_config.base_dir) / self.output_config.scored_dir
    
    def get_figures_path(self) -> Path:
        """Get the path for figures."""
        return Path(self.output_config.base_dir) / self.output_config.figures_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'dataset': {
                'split': self.dataset_config.split,
                'n_problems': self.dataset_config.n_problems,
                'difficulty': self.dataset_config.difficulty,
                'sample_types': self.dataset_config.sample_types
            },
            'models': {
                'models': self.model_config.models,
                'max_workers': self.model_config.max_workers
            },
            'generation': {
                'max_tokens': self.model_config.max_tokens,
                'temperature': self.model_config.temperature,
                'timeout': self.model_config.timeout
            },
            'output': {
                'base_dir': self.output_config.base_dir,
                'generation_dir': self.output_config.generation_dir,
                'scored_dir': self.output_config.scored_dir,
                'figures_dir': self.output_config.figures_dir
            }
        }
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.dataset_config.n_problems < 1:
            raise ValueError("n_problems must be at least 1")
        
        if self.dataset_config.split not in ["train", "eval", "test"]:
            raise ValueError(f"Invalid split: {self.dataset_config.split}")
        
        if self.model_config.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        if self.model_config.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        
        if self.model_config.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        
        if not 0 <= self.model_config.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        # Create output directories if they don't exist
        for path_func in [self.get_generation_path, self.get_scored_path, self.get_figures_path]:
            path = path_func()
            path.mkdir(parents=True, exist_ok=True)