"""Tests for data loading modules."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.patent_loader import PatentLoader
from src.data.onet_loader import ONetLoader


class TestPatentLoader:
    """Test PatentLoader class."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration."""
        return {
            'processing': {
                'ai_topic_filter': 1
            }
        }
    
    @pytest.fixture
    def sample_patents_df(self):
        """Sample patent DataFrame."""
        return pd.DataFrame({
            'application_id': [1, 2, 3],
            'application_title': ['AI Patent 1', 'AI Patent 2', 'Non-AI Patent'],
            'topic_label': ['Machine Learning', 'Neural Networks', 'Manufacturing'],
            'year': [2021.0, 2021.0, 2022.0]
        })
    
    def test_load_patents(self, config, sample_patents_df, tmp_path):
        """Test loading patents from file."""
        # Save sample data to temp file
        filepath = tmp_path / "patents.csv"
        sample_patents_df.to_csv(filepath, index=False)
        
        loader = PatentLoader(config)
        df = loader.load_patents(str(filepath))
        
        assert len(df) == 3
        assert 'application_title' in df.columns
    
    def test_filter_ai_patents(self, config, sample_patents_df):
        """Test filtering AI patents."""
        categories_df = pd.DataFrame({
            'AI topic': [1, 1, 0],
            'Topics': ['Machine Learning', 'Neural Networks', 'Manufacturing']
        })
        
        loader = PatentLoader(config)
        loader.patents_df = sample_patents_df
        
        ai_patents = loader.filter_ai_patents(categories_df)
        
        assert len(ai_patents) == 2
        assert all(title.startswith('AI') for title in ai_patents['application_title'])
    
    def test_get_patents_by_year(self, config, sample_patents_df):
        """Test getting patents by year."""
        loader = PatentLoader(config)
        loader.ai_patents_df = sample_patents_df
        
        year_patents = loader.get_patents_by_year(2021)
        
        assert len(year_patents) == 2
        assert all(year == 2021.0 for year in year_patents['year'])


class TestONetLoader:
    """Test ONetLoader class."""
    
    @pytest.fixture
    def config(self):
        """Sample configuration."""
        return {}
    
    @pytest.fixture
    def sample_onet_df(self):
        """Sample O*NET DataFrame."""
        return pd.DataFrame({
            'Task ID': [1, 2, 3],
            'Task': ['Task 1', 'Task 2', 'Task 1'],
            'Title': ['Occupation 1', 'Occupation 2', 'Occupation 3']
        })
    
    def test_load_task_ratings(self, config, sample_onet_df, tmp_path):
        """Test loading O*NET task ratings."""
        filepath = tmp_path / "tasks.xlsx"
        sample_onet_df.to_excel(filepath, index=False)
        
        loader = ONetLoader(config)
        df = loader.load_task_ratings(str(filepath))
        
        assert len(df) == 3
        assert 'Task' in df.columns
    
    def test_get_unique_tasks(self, config, sample_onet_df):
        """Test getting unique tasks."""
        loader = ONetLoader(config)
        loader.tasks_df = sample_onet_df
        
        unique_tasks = loader.get_unique_tasks()
        
        assert len(unique_tasks) == 2
        assert 'Task 1' in unique_tasks
        assert 'Task 2' in unique_tasks
    
    def test_add_matching_results(self, config, sample_onet_df):
        """Test adding matching results."""
        loader = ONetLoader(config)
        loader.tasks_df = sample_onet_df
        
        neighbors = [[1, 2], [3, 4], [5, 6]]
        distances = [[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]]
        
        loader.add_matching_results(2021, neighbors, distances)
        
        assert 'neighbors_2021' in loader.tasks_df.columns
        assert 'distances_2021' in loader.tasks_df.columns