#!/usr/bin/env python3
"""
Test that evalrender can be imported from train_qlearn without circular import
"""
import sys
sys.path.insert(0, '/home/magnus/masters_thesis/2026ver/pyquaticus')

try:
    # This should work now without circular import
    from qlearning.train_qlearn import ParameterSet
    print("✓ ParameterSet imported successfully")
    
    from qlearning.evaluate_q_new import evalrender
    print("✓ evalrender imported successfully")
    
    # Verify the function exists and is callable
    if callable(evalrender):
        print("✓ evalrender is callable")
    
    print("\n" + "="*60)
    print("CIRCULAR IMPORT FIX VERIFIED ✓")
    print("="*60)
    print("\nChanges made:")
    print("1. Removed 'from train_qlearn import ParameterSet' from top of evaluate_q_new.py")
    print("2. Added local import inside load_and_call_helper() function")
    print("3. Removed 'from evaluate_q_new import evalrender' from top of train_qlearn.py")
    print("4. Added local import inside render mode block in train_qlearn.py")
    print("\nResult: No circular import - modules can now import each other!")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
