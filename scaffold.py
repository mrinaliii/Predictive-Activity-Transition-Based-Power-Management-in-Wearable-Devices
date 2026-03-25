"""
Scaffold: Create complete project folder structure and empty placeholder files.
Run: python scaffold.py
"""

import os
from pathlib import Path

def create_scaffold():
    """Build project directory structure and empty files."""
    
    base_path = Path(__file__).parent
    
    # Define folder structure
    folders = [
        "data/raw",
        "data/processed",
        "src",
        "models/saved_models",
        "results/graphs",
        "results/metrics",
        "results/logs",
        "notebooks",
        "config"
    ]
    
    # Create all directories
    for folder in folders:
        folder_path = base_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder_path.relative_to(base_path)}")
    
    # Define files to create (empty placeholders)
    files = [
        "src/data_loader.py",
        "src/preprocess.py",
        "src/feature_engineering.py",
        "src/model.py",
        "src/train.py",
        "src/evaluate.py",
        "src/energy_simulation.py",
        "config/config.yaml",
        "main.py",
        "requirements.txt",
        "README.md"
    ]
    
    # Create all empty files
    for file_path in files:
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Only create if doesn't exist
        if not full_path.exists():
            full_path.touch()
            print(f"✓ Created: {full_path.relative_to(base_path)}")
        else:
            print(f"~ Already exists: {full_path.relative_to(base_path)}")
    
    print("\n" + "="*60)
    print("✓ Scaffold complete! Project structure is ready.")
    print("="*60)
    
    # Print ASCII tree
    print("\nProject structure:")
    print(f"{base_path.name}/")
    print("├── data/")
    print("│   ├── raw/")
    print("│   └── processed/")
    print("├── src/")
    print("│   ├── data_loader.py")
    print("│   ├── preprocess.py")
    print("│   ├── feature_engineering.py")
    print("│   ├── model.py")
    print("│   ├── train.py")
    print("│   ├── evaluate.py")
    print("│   └── energy_simulation.py")
    print("├── models/")
    print("│   └── saved_models/")
    print("├── results/")
    print("│   ├── graphs/")
    print("│   ├── metrics/")
    print("│   └── logs/")
    print("├── notebooks/")
    print("├── config/")
    print("│   └── config.yaml")
    print("├── main.py")
    print("├── requirements.txt")
    print("└── README.md")

if __name__ == "__main__":
    create_scaffold()
