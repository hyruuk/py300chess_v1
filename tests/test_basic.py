"""
Basic tests to verify core components work correctly.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import chess
from game.game_manager import ChessGameManager
from game.move_selector import P300MoveSelector
from classification.p300_detector import P300Detector
from classification.calibration import CalibrationSession
from processing.preprocessing import EEGPreprocessor
from processing.feature_extraction import P300FeatureExtractor


def test_chess_game_manager():
    """Test chess game manager."""
    print("Testing ChessGameManager...")

    game = ChessGameManager()

    # Test initial position
    assert game.board.fen() == chess.STARTING_FEN
    assert len(game.get_legal_moves()) == 20  # 20 legal moves in starting position

    # Test making a move
    move = chess.Move.from_uci("e2e4")
    assert game.make_move(move) == True

    # Test illegal move
    illegal_move = chess.Move.from_uci("e2e4")  # Same square
    assert game.make_move(illegal_move) == False

    # Test undo
    undone = game.undo_move()
    assert undone == move
    assert game.board.fen() == chess.STARTING_FEN

    print("✓ ChessGameManager tests passed")


def test_move_selector():
    """Test P300 move selector."""
    print("Testing P300MoveSelector...")

    selector = P300MoveSelector(
        flash_duration=0.15,
        isi=0.25,
        n_repetitions=5
    )

    # Test flash sequence creation
    flash_seq = selector.create_flash_sequence()
    assert len(flash_seq) == 5 * 16  # 5 reps * (8 rows + 8 cols)

    # Test square selection
    p300_scores = np.random.rand(len(flash_seq))
    square, confidence = selector.determine_selected_square(flash_seq, p300_scores)

    assert 0 <= square < 64
    assert 'row_confidence' in confidence
    assert 'col_confidence' in confidence

    print("✓ P300MoveSelector tests passed")


def test_preprocessing():
    """Test signal preprocessing."""
    print("Testing EEGPreprocessor...")

    preprocessor = EEGPreprocessor(
        sampling_rate=250,
        bandpass=(0.1, 20),
        notch_freq=50
    )

    # Create synthetic EEG data
    n_samples = 1000
    n_channels = 8
    data = np.random.randn(n_samples, n_channels)

    # Test filtering
    filtered = preprocessor.filter(data)
    assert filtered.shape == data.shape

    # Test epoching
    timestamps = np.arange(n_samples) / 250.0
    event_time = 2.0
    epoch = preprocessor.epoch_data(data, timestamps, event_time, -0.2, 0.8)

    assert epoch is not None
    assert epoch.shape[1] == n_channels

    # Test baseline correction
    baseline_window = preprocessor.get_baseline_window(-0.2, 0.0)
    corrected = preprocessor.baseline_correct(epoch, baseline_window)
    assert corrected.shape == epoch.shape

    print("✓ EEGPreprocessor tests passed")


def test_p300_detector():
    """Test P300 detector."""
    print("Testing P300Detector...")

    detector = P300Detector(method='LDA')

    # Create synthetic training data
    n_epochs = 100
    n_samples = 200
    n_channels = 8

    X = np.random.randn(n_epochs, n_samples, n_channels)
    y = np.random.randint(0, 2, n_epochs)

    # Train detector
    metrics = detector.train(X, y)

    assert detector.is_trained
    assert 'cv_accuracy' in metrics

    # Test prediction
    test_epoch = np.random.randn(n_samples, n_channels)
    prediction, confidence = detector.predict(test_epoch)

    assert prediction in [0, 1]
    assert 0 <= confidence <= 1

    print(f"✓ P300Detector tests passed (CV accuracy: {metrics['cv_accuracy']:.3f})")


def test_calibration_session():
    """Test calibration session."""
    print("Testing CalibrationSession...")

    calibration = CalibrationSession(
        n_targets=8,
        n_trials_per_target=5,
        n_nontargets_per_target=3,
        flash_duration=0.1,
        isi=0.3
    )

    # Check trial generation
    assert len(calibration.trials) == 8 * 5  # n_targets * n_trials_per_target

    # Test adding epochs
    epoch = np.random.randn(100, 8)
    calibration.add_epoch(epoch, is_target=True)

    X, y = calibration.get_training_data()
    assert len(X) == 1
    assert len(y) == 1
    assert y[0] == 1

    print("✓ CalibrationSession tests passed")


def test_feature_extraction():
    """Test feature extraction."""
    print("Testing P300FeatureExtractor...")

    extractor = P300FeatureExtractor(
        target_channels=[0, 1, 2, 3],
        sampling_rate=250,
        downsample_factor=4
    )

    # Create synthetic epoch
    n_samples = 200
    n_channels = 8
    epoch = np.random.randn(n_samples, n_channels)

    # Test temporal features
    features = extractor.extract_temporal_features(epoch)
    assert len(features) > 0

    # Test spatial features
    spatial_filters = np.random.randn(n_channels, 4)
    spatial_features = extractor.extract_spatial_features(epoch, spatial_filters)
    assert len(spatial_features) > 0

    print("✓ P300FeatureExtractor tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running P300 Chess BCI Tests")
    print("=" * 60)
    print()

    try:
        test_chess_game_manager()
        test_move_selector()
        test_preprocessing()
        test_feature_extraction()
        test_calibration_session()
        test_p300_detector()

        print()
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
