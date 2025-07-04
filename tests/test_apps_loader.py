"""
Test APPS dataset loader functionality.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from apps.load_apps_dataset import APPSDatasetLoader


def test_apps_loader_initialization():
    """Test APPSDatasetLoader initialization."""
    loader = APPSDatasetLoader()
    assert loader.data_dir.exists()


def test_create_prompt():
    """Test prompt creation."""
    loader = APPSDatasetLoader()
    
    problem = {
        'id': 'test_001',
        'title': 'Test Problem',
        'question': 'Write a function that adds two numbers.',
        'starter_code': 'def add(a, b):\n    pass',
        'difficulty': 'easy'
    }
    
    prompt = loader.create_prompt(problem)
    assert 'Test Problem' in prompt
    assert 'Write a function that adds two numbers' in prompt
    assert 'def add(a, b):' in prompt


def test_create_prompt_without_starter_code():
    """Test prompt creation without starter code."""
    loader = APPSDatasetLoader()
    
    problem = {
        'id': 'test_002',
        'title': 'Test Problem 2',
        'question': 'Write a function that multiplies two numbers.',
        'starter_code': '',
        'difficulty': 'medium'
    }
    
    prompt = loader.create_prompt(problem, include_starter_code=False)
    assert 'Test Problem 2' in prompt
    assert 'Write a function that multiplies two numbers' in prompt
    assert 'def add(a, b):' not in prompt


if __name__ == "__main__":
    pytest.main([__file__]) 