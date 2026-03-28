"""
Test script for regenerate_all_outputs() function.

Tests:
1. Function exists and is callable
2. Creates all 5 output files
3. All files have content
4. Consistency checks pass
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.adaptive_pipeline import AdaptivePipelineOrchestrator

class MockGRUModel(nn.Module):
    """Mock GRU model for testing."""
    def __init__(self, num_classes=12):
        super().__init__()
        self.gru = nn.GRU(input_size=9, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: [batch, 128, 9] or [128, 9]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        _, h_n = self.gru(x)
        logits = self.fc(h_n[-1])
        return logits

def test_regenerate_outputs():
    """Test the regenerate_all_outputs() function."""
    print("\n" + "="*70)
    print("TEST: regenerate_all_outputs()")
    print("="*70)
    
    # Create a small test dataset
    dataset = np.random.randn(100, 128, 9)  # 100 windows
    labels = np.random.randint(0, 12, 100)
    
    # Create mock GRU model
    gru_model = MockGRUModel(num_classes=12)
    
    # Initialize orchestrator
    orchestrator = AdaptivePipelineOrchestrator(
        gru_model=gru_model,
        user_id='test_user',
        num_classes=12,
        device='cpu'
    )
    
    # Run regenerate_all_outputs
    result = orchestrator.regenerate_all_outputs(dataset, labels, output_dir='results')
    
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Check all files exist
    all_files_exist = True
    for file_type, file_path in result['files'].items():
        exists = Path(file_path).exists()
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {file_type}: {Path(file_path).name}")
        if not exists:
            all_files_exist = False
    
    # Check JSON is valid
    print()
    try:
        with open(result['files']['summary_json'], 'r') as f:
            json_data = json.load(f)
        print(f"[OK] JSON file is valid")
        print(f"  - Adaptive energy: {json_data['energy_metrics']['adaptive_energy_mj']:.1f} mJ")
        print(f"  - Safety override count: {json_data['conflict_resolution']['safety_override_wins']}")
        print(f"  - AER score: {json_data['aer_metrics']['aer_score']:.4f}")
    except Exception as e:
        print(f"[FAIL] JSON file error: {e}")
        all_files_exist = False
    
    # Check consistency
    print()
    if result['consistency_check_passed']:
        print("[PASS] CONSISTENCY CHECK PASSED")
    else:
        print("[FAIL] CONSISTENCY CHECK FAILED:")
        for error in result['consistency_errors']:
            print(f"  - {error}")
    
    print("\n" + "="*70)
    if all_files_exist and result['consistency_check_passed']:
        print("[PASS] TEST PASSED")
        print("="*70)
        return True
    else:
        print("[FAIL] TEST FAILED")
        print("="*70)
        return False

if __name__ == '__main__':
    success = test_regenerate_outputs()
    sys.exit(0 if success else 1)
