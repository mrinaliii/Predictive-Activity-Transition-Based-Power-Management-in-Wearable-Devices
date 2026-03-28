"""
Standalone test for ConflictResolver logic without torch dependency.
Tests the core conflict resolution algorithm.
"""

from datetime import datetime
from typing import Dict, Any
from collections import defaultdict


class ConflictResolver:
    """
    Resolves conflicts between multiple control signals using a priority hierarchy.
    
    Priority levels (highest to lowest):
        Level 1: Safety Override (Z=1) — always wins, no exceptions
        Level 2: Transition Watchdog (T > threshold) — increases power
        Level 3: Confidence Tier — baseline power decision
    """
    
    def __init__(self) -> None:
        """Initialize ConflictResolver with empty conflict tracking."""
        self.conflict_counts: Dict[str, int] = {
            'safety_override_wins': 0,
            'transition_overrides_confidence': 0,
            'confidence_only': 0
        }
    
    def resolve(self, 
                safety_state: Dict[str, Any], 
                transition_state: Dict[str, Any], 
                confidence_state: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between safety, transition, and confidence signals."""
        # Priority Level 1: Safety Override
        if safety_state['override_active']:
            self.conflict_counts['safety_override_wins'] += 1
            return {
                'sampling_rate': safety_state['recommended_sampling_rate'],
                'active_axes': list(range(1, 10)),  # All 9 axes
                'reason': 'safety_override',
                'priority_level': 1
            }
        
        # Priority Level 2: Transition Watchdog
        if transition_state.get('transition_detected', False):
            self.conflict_counts['transition_overrides_confidence'] += 1
            combined_rate = max(
                transition_state.get('transition_rate', 100),
                confidence_state['sampling_rate']
            )
            combined_axes = sorted(list(set(
                transition_state.get('transition_axes', list(range(1, 10))) +
                confidence_state['active_axes']
            )))
            return {
                'sampling_rate': combined_rate,
                'active_axes': combined_axes,
                'reason': 'transition_conflict_resolved',
                'priority_level': 2
            }
        
        # Priority Level 3: Confidence Tier
        self.conflict_counts['confidence_only'] += 1
        return {
            'sampling_rate': confidence_state['sampling_rate'],
            'active_axes': confidence_state['active_axes'],
            'reason': 'confidence_only',
            'priority_level': 3
        }
    
    def log_decision(self, decision: Dict[str, Any]) -> None:
        """Log a conflict resolution decision with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(
            f"[{timestamp}] Priority L{decision['priority_level']}: "
            f"{decision['reason']} → {decision['sampling_rate']} Hz, "
            f"axes={decision['active_axes']}"
        )
    
    def __repr__(self) -> str:
        """Return formatted priority hierarchy as string."""
        hierarchy = (
            "ConflictResolver Priority Hierarchy:\n"
            "  L1 (HIGHEST): Safety Override (Z=1) → always wins\n"
            "  L2 (MEDIUM):  Transition Watchdog (T > threshold) → increases power\n"
            "  L3 (LOWEST):  Confidence Tier → baseline power\n"
            f"  Conflicts resolved: {sum(self.conflict_counts.values())}\n"
            f"    - Safety wins: {self.conflict_counts['safety_override_wins']}\n"
            f"    - Transition wins: {self.conflict_counts['transition_overrides_confidence']}\n"
            f"    - Confidence only: {self.conflict_counts['confidence_only']}"
        )
        return hierarchy


def run_tests():
    """Run comprehensive conflict resolver tests."""
    
    print("\n" + "="*80)
    print("CONFLICT RESOLVER VERIFICATION TEST SUITE")
    print("="*80)
    
    resolver = ConflictResolver()
    print(f"\n{resolver}\n")
    
    # Test Case 1: Safety Override Wins (Priority L1)
    print("\n" + "-"*80)
    print("TEST 1: Safety Override Active (L1 - HIGHEST PRIORITY)")
    print("-"*80)
    
    safety_state_active = {
        'override_active': True,
        'reason': 'Fall_Detected_IMU',
        'recommended_sampling_rate': 100
    }
    transition_state = {
        'transition_detected': False,
        'transition_rate': 100,
        'transition_axes': list(range(1, 10))
    }
    confidence_state = {
        'sampling_rate': 25,
        'active_axes': [1, 2, 3],
        'confidence': 0.95,
        'tier': 'high'
    }
    
    result1 = resolver.resolve(safety_state_active, transition_state, confidence_state)
    print(f"Input:  Safety={safety_state_active['override_active']}, "
          f"Transition={transition_state['transition_detected']}")
    print(f"Output: {result1}")
    
    assert result1['reason'] == 'safety_override', "❌ FAILED: Wrong reason"
    assert result1['sampling_rate'] == 100, "❌ FAILED: Wrong sampling rate"
    assert result1['priority_level'] == 1, "❌ FAILED: Wrong priority level"
    assert result1['active_axes'] == list(range(1, 10)), "❌ FAILED: Wrong axes"
    print("✓ PASSED: Safety override L1 priority correctly enforced")
    
    # Test Case 2: Transition Overrides Confidence (Priority L2)
    print("\n" + "-"*80)
    print("TEST 2: Transition Detected + Confidence (L2 - MEDIUM PRIORITY)")
    print("-"*80)
    
    safety_state_inactive = {
        'override_active': False,
        'reason': None,
        'recommended_sampling_rate': 0
    }
    transition_state_active = {
        'transition_detected': True,
        'transition_rate': 100,
        'transition_axes': list(range(1, 10))
    }
    confidence_state_medium = {
        'sampling_rate': 50,
        'active_axes': [1, 2, 3, 5],
        'confidence': 0.70,
        'tier': 'medium'
    }
    
    result2 = resolver.resolve(safety_state_inactive, transition_state_active, confidence_state_medium)
    print(f"Input:  Safety={safety_state_inactive['override_active']}, "
          f"Transition={transition_state_active['transition_detected']}, "
          f"Conf_rate={confidence_state_medium['sampling_rate']}")
    print(f"Output: {result2}")
    
    assert result2['reason'] == 'transition_conflict_resolved', "❌ FAILED: Wrong reason"
    assert result2['sampling_rate'] == 100, "❌ FAILED: sampling_rate should be max(100,50)=100"
    assert result2['priority_level'] == 2, "❌ FAILED: Wrong priority level"
    assert set(result2['active_axes']) == set(range(1, 10)), "❌ FAILED: Union should have all axes"
    print("✓ PASSED: Transition L2 priority correctly resolved")
    
    # Test Case 3: Confidence Only (Priority L3)
    print("\n" + "-"*80)
    print("TEST 3: No Safety, No Transition, Use Confidence (L3 - LOWEST PRIORITY)")
    print("-"*80)
    
    transition_state_inactive = {
        'transition_detected': False,
        'transition_rate': 100,
        'transition_axes': list(range(1, 10))
    }
    confidence_state_low = {
        'sampling_rate': 100,
        'active_axes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'confidence': 0.45,
        'tier': 'low'
    }
    
    result3 = resolver.resolve(safety_state_inactive, transition_state_inactive, confidence_state_low)
    print(f"Input:  Safety={safety_state_inactive['override_active']}, "
          f"Transition={transition_state_inactive['transition_detected']}")
    print(f"Output: {result3}")
    
    assert result3['reason'] == 'confidence_only', "❌ FAILED: Wrong reason"
    assert result3['sampling_rate'] == 100, "❌ FAILED: Wrong sampling rate"
    assert result3['priority_level'] == 3, "❌ FAILED: Wrong priority level"
    print("✓ PASSED: Confidence L3 priority correctly applied")
    
    # Test Case 4: Transition with Lower Confidence Rate
    print("\n" + "-"*80)
    print("TEST 4: Transition with Low Confidence (max() resolution)")
    print("-"*80)
    
    confidence_state_high = {
        'sampling_rate': 25,  # High confidence, LOW rate
        'active_axes': [1, 2, 3],
        'confidence': 0.92,
        'tier': 'high'
    }
    
    result4 = resolver.resolve(safety_state_inactive, transition_state_active, confidence_state_high)
    print(f"Input:  Transition=True, Transition_rate=100, Confidence_rate=25")
    print(f"Output: {result4}")
    
    assert result4['sampling_rate'] == 100, "❌ FAILED: Should use max(100, 25) = 100"
    assert result4['reason'] == 'transition_conflict_resolved', "❌ FAILED: Wrong reason"
    print("✓ PASSED: Transition correctly overrides lower confidence rate")
    
    # Test Case 5: Transition with Higher Confidence Rate
    print("\n" + "-"*80)
    print("TEST 5: Transition with High Confidence (max() with reversal)")
    print("-"*80)
    
    confidence_state_ultra_low = {
        'sampling_rate': 100,  # Low confidence, HIGH rate (unusual but valid)
        'active_axes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'confidence': 0.35,
        'tier': 'low'
    }
    
    transition_state_lower = {
        'transition_detected': True,
        'transition_rate': 50,  # Hypothetically lower
        'transition_axes': [1, 2, 3, 4, 5]
    }
    
    result5 = resolver.resolve(safety_state_inactive, transition_state_lower, confidence_state_ultra_low)
    print(f"Input:  Transition=True, Transition_rate=50, Confidence_rate=100")
    print(f"Output: {result5}")
    
    assert result5['sampling_rate'] == 100, "❌ FAILED: Should use max(50, 100) = 100"
    print("✓ PASSED: Correctly uses max() even when confidence rate is higher")
    
    # Summary Statistics
    print("\n" + "-"*80)
    print("CONFLICT RESOLUTION STATISTICS")
    print("-"*80)
    print(f"Safety Override Wins (L1): {resolver.conflict_counts['safety_override_wins']}")
    print(f"Transition Overrides (L2): {resolver.conflict_counts['transition_overrides_confidence']}")
    print(f"Confidence Only (L3):      {resolver.conflict_counts['confidence_only']}")
    print(f"Total Resolutions:         {sum(resolver.conflict_counts.values())}")
    
    expected_totals = {
        'safety_override_wins': 1,
        'transition_overrides_confidence': 3,  # Tests 2, 4, 5
        'confidence_only': 1
    }
    
    for key, expected in expected_totals.items():
        actual = resolver.conflict_counts[key]
        assert actual == expected, f"❌ FAILED: {key} should be {expected}, got {actual}"
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED - ConflictResolver working correctly")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_tests()
