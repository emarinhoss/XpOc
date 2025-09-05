#!/usr/bin/env python
"""Quick fix for data issues."""
import pandas as pd
from pathlib import Path

def create_sample_data():
    """Create all necessary sample data files."""
    
    # Create directories
    for dir_path in ['data/raw/patents', 'data/raw/onet', 'data/raw/categories', 
                     'data/processed', 'data/processed/matches']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create sample patents
    patents_data = {
        'application_id': range(1, 101),
        'application_title': [f'AI Patent {i}: {tech}' 
                            for i in range(1, 101) 
                            for tech in ['Machine Learning System'] * 1][:100],
        'filing_date': ['2021-01-01'] * 33 + ['2022-01-01'] * 33 + ['2023-01-01'] * 34,
        'year': [2021.0] * 33 + [2022.0] * 33 + [2023.0] * 34,
        'topic_label': ['Machine Learning'] * 50 + ['Neural Networks'] * 50
    }
    
    patents_df = pd.DataFrame(patents_data)
    patents_df.to_csv('data/processed/patents.csv', index=False)
    print("✓ Created patents.csv")
    
    # Create sample O*NET data
    onet_data = {
        'Task ID': range(1, 21),
        'Task': [
            'Analyze complex data sets using statistical methods',
            'Develop predictive models using machine learning',
            'Design and implement database systems',
            'Create automated data processing pipelines',
            'Build and deploy AI applications',
            'Perform data visualization and reporting',
            'Implement natural language processing systems',
            'Develop computer vision algorithms',
            'Create recommendation engines',
            'Build chatbot and conversational AI systems',
            'Implement deep learning neural networks',
            'Perform feature engineering and selection',
            'Optimize algorithm performance and efficiency',
            'Conduct A/B testing and experimentation',
            'Deploy models to production environments',
            'Monitor and maintain ML model performance',
            'Collaborate with stakeholders on requirements',
            'Document technical processes and solutions',
            'Perform code reviews and quality assurance',
            'Research and implement new AI techniques'
        ],
        'Title': ['Data Scientist'] * 5 + ['ML Engineer'] * 5 + 
                ['AI Researcher'] * 5 + ['Software Engineer'] * 5,
        'O*NET-SOC Code': ['15-2051.00'] * 5 + ['15-1299.08'] * 5 + 
                         ['15-1299.09'] * 5 + ['15-1252.00'] * 5
    }
    
    onet_df = pd.DataFrame(onet_data)
    onet_df.to_csv('data/processed/onet_tasks.csv', index=False)
    print("✓ Created onet_tasks.csv")
    
    # Create AI categories
    categories_data = {
        'AI topic': [1, 1, 0],
        'Topics': ['Machine Learning', 'Neural Networks', 'Traditional Software']
    }
    
    categories_df = pd.DataFrame(categories_data)
    Path('data/raw/categories').mkdir(parents=True, exist_ok=True)
    categories_df.to_excel('data/raw/categories/patent_categories.xlsx', index=False)
    print("✓ Created patent_categories.xlsx")
    
    print("\n✓ All sample data created successfully!")

if __name__ == "__main__":
    create_sample_data()