"""
Results persistence layer for handling file I/O operations.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ResultsPersistence:
    """Handles persistence of generation and evaluation results."""
    
    def __init__(self, base_dir: str = "data"):
        """Initialize persistence layer.
        
        Args:
            base_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_dir)
        self.generation_dir = self.base_dir / "generation_outputs"
        self.scored_dir = self.base_dir / "scored_outputs"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure output directories exist."""
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.scored_dir.mkdir(parents=True, exist_ok=True)
    
    def save_generation_results(self, results: Dict, metadata: Dict) -> Path:
        """Save generation results to JSON file.
        
        Args:
            results: Model generation results
            metadata: Metadata about the generation run
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Create filename based on metadata
        split = metadata.get('split', 'unknown')
        n_problems = metadata.get('n_problems', 0)
        n_models = len(metadata.get('models', []))
        
        filename = f"{timestamp_str}_{split}_{n_problems}problems_{n_models}models_outputs.json"
        filepath = self.generation_dir / filename
        
        # Create data structure
        data = {
            "metadata": {
                **metadata,
                "timestamp": timestamp.isoformat()
            },
            "results": results
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def save_scored_results(self, results: Dict, summary: Dict, metadata: Dict) -> Path:
        """Save scored evaluation results to JSON file.
        
        Args:
            results: Scored model results
            summary: Summary statistics
            metadata: Metadata about the evaluation
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Create filename based on metadata
        split = metadata.get('split', 'unknown')
        n_problems = metadata.get('n_problems', 0)
        n_models = len(metadata.get('models', []))
        
        filename = f"{timestamp_str}_evalproblems_{n_problems}problemsmodels_scored.json"
        filepath = self.scored_dir / filename
        
        # Create data structure
        data = {
            "metadata": {
                **metadata,
                "timestamp": timestamp.isoformat()
            },
            "summary": summary,
            "results": results
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load_generation_results(self, filepath: str) -> Dict:
        """Load generation results from file.
        
        Args:
            filepath: Path to generation results file
            
        Returns:
            Dictionary with generation results
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def find_latest_generation_file(self) -> Path:
        """Find the most recent generation output file.
        
        Returns:
            Path to the latest file or None if no files exist
        """
        files = sorted(self.generation_dir.glob("*_outputs.json"), reverse=True)
        return files[0] if files else None
    
    def find_latest_scored_file(self) -> Path:
        """Find the most recent scored output file.
        
        Returns:
            Path to the latest file or None if no files exist
        """
        files = sorted(self.scored_dir.glob("*_scored.json"), reverse=True)
        return files[0] if files else None