"""
Test that all modules can be imported without errors.
This test doesn't require any dependencies to be installed.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules import successfully."""
    print("Testing imports...")

    try:
        # Test acquisition modules
        from acquisition import lsl_client, data_buffer
        print("✓ acquisition modules")

        # Test processing modules
        from processing import preprocessing, feature_extraction
        print("✓ processing modules")

        # Test classification modules
        from classification import calibration, p300_detector
        print("✓ classification modules")

        # Test game modules (renamed from chess)
        from game import game_manager, move_selector, keyboard_selector, engine_interface
        print("✓ game modules")

        # Test GUI modules
        from gui import main_window, chess_board, flash_controller
        print("✓ gui modules")

        print("\n✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
