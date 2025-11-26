# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

P300 Chess BCI is a brain-computer interface application that enables locked-in syndrome patients to play chess using P300 event-related potentials detected from EEG signals. The system uses a P300-speller paradigm with row/column flashing for move selection.

## Common Commands

### Running the Application

```bash
# Run with demo mode (keyboard controls, no EEG required)
python src/main.py

# Run with custom config
python src/main.py --config path/to/settings.yaml

# Run in virtual environment
source env/bin/activate  # or venv/bin/activate
python src/main.py
```

### Testing

```bash
# Run all basic tests
python tests/test_basic.py

# Run import tests
python tests/test_imports.py

# Run syntax validation
python tests/test_syntax.py

# Test specific component (from project root)
python -c "from src.game.game_manager import ChessGameManager; print('Import successful')"
```

### Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install Stockfish chess engine (required)
sudo apt-get install stockfish  # Ubuntu/Debian
brew install stockfish          # macOS
```

## Architecture

### Core Data Flow

The application follows a pipeline architecture for BCI processing:

```
EEG Stream → Buffer → Preprocessing → Epoching → Feature Extraction → Classification → Move Selection → Chess Engine
```

### Module Organization

**src/acquisition/**: EEG data acquisition
- `lsl_client.py`: Connects to Lab Streaming Layer (LSL) EEG streams
- `data_buffer.py`: Circular buffer for continuous EEG data storage
- `simulated_eeg.py`: Generates synthetic EEG for demo/testing mode

**src/processing/**: Signal processing pipeline
- `preprocessing.py`: Bandpass filtering (0.1-20 Hz), notch filtering, epoching, baseline correction
- `feature_extraction.py`: Temporal and spatial feature extraction from EEG epochs

**src/classification/**: P300 detection and training
- `calibration.py`: Calibration session management, generates trial sequences
- `p300_detector.py`: P300 classifier (LDA, xDAWN+LDA), training and prediction

**src/game/**: Chess game logic and move selection
- `game_manager.py`: Chess game state using python-chess library
- `move_selector.py`: P300-based move selection using row/column flashing paradigm
- `keyboard_selector.py`: Keyboard control for demo mode (arrow keys + spacebar)
- `engine_interface.py`: Stockfish chess engine integration
- `live_p300_engine.py`: Real-time P300 move selection engine

**src/gui/**: Pygame-based user interface
- `main_window.py`: Main application window and event loop
- `chess_board.py`: Chess board rendering
- `flash_controller.py`: Visual stimulus flash timing and control
- `eeg_visualizer.py`: Real-time EEG signal visualization

**src/main.py**: Application entry point, coordinates all components

### Key Architectural Patterns

**Two-Step Move Selection Paradigm**:
1. Flash rows (1-8) and columns (a-h) to select origin square
2. Flash again to select destination square
3. P300 response indicates attended stimulus
4. Intersection of selected row/column determines square

**Calibration-Then-Play Workflow**:
- Calibration phase collects labeled training data (targets vs non-targets)
- Trains classifier on participant-specific P300 responses
- Game phase uses trained classifier for real-time move selection
- Calibration data saved to `calibrations/` directory with timestamps

**Demo Mode Architecture**:
- `config/settings.yaml`: Set `demo.enabled: true` and `demo.keyboard_control: true`
- `eeg.use_simulated: true` enables simulated EEG stream instead of LSL
- Keyboard mode bypasses P300 detection, uses arrow keys for direct selection
- Allows testing full game flow without EEG hardware

### Configuration System

All settings managed in `config/settings.yaml`:
- **eeg**: Stream settings, channels, sampling rate, simulated vs real
- **processing**: Filter parameters, epoch windows, baseline correction
- **p300**: Classifier type (LDA, xDAWN_LDA), target channels
- **calibration**: Trial counts, flash timing
- **chess**: Engine settings, flash parameters, repetitions
- **demo**: Enable keyboard mode, show legal moves
- **gui**: Screen size, colors, board rendering

### Data Persistence

**Calibration Files**: `calibrations/calibration_YYYY-MM-DD_HH-MM-SS_StreamName.pkl`
- Contains trained classifier, training data, and metadata
- Pickle format, ~8MB per file
- Auto-generated timestamp naming

**Session Logs**: Application logs stored in `logs/` (if configured)

## Development Notes

### EEG Signal Processing

The preprocessing pipeline expects:
- Sampling rate: 250 Hz (configurable)
- Channels: ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'C3', 'C4']
- P300 target channels: ['Pz', 'Cz', 'P3', 'P4'] (central/parietal)
- Epoch window: -0.2s to 0.8s relative to stimulus onset
- Baseline: -0.2s to 0s

### Chess Engine Integration

Stockfish integration via `python-chess` library:
- UCI protocol communication
- Configurable skill level (0-20) and search depth
- Engine path auto-detection or manual specification
- Move validation through `python-chess.Board`

### Testing Without Hardware

To develop without EEG hardware:
1. Set `eeg.use_simulated: true` in config
2. Set `demo.enabled: true` and `demo.keyboard_control: true`
3. `SimulatedEEGStream` generates synthetic signals with realistic P300 patterns
4. Keyboard selector allows direct move input with arrow keys

### Adding New Classifiers

To implement a new classification method:
1. Edit `src/classification/p300_detector.py`
2. Add method to `P300Detector.train()` and `P300Detector.predict()`
3. Use scikit-learn compatible interface
4. Update `config/settings.yaml` with new classifier name

### Flash Sequence Timing

Critical timing parameters:
- `flash_duration`: How long each stimulus flashes (default: 0.1s)
- `isi`: Inter-stimulus interval between flashes (default: 0.2s)
- `n_repetitions`: How many times full sequence repeats (default: 5)
- More repetitions = higher accuracy but slower selection

### Common Issues

**LSL Connection Failures**:
- Check `eeg.stream_name` matches actual LSL stream
- Verify EEG amplifier is streaming
- Use `pylsl.resolve_streams()` to list available streams

**Import Errors**:
- Ensure virtual environment is activated
- Check `sys.path.insert(0, ...)` in files that need src/ imports
- All src/ modules use relative imports within package

**Stockfish Not Found**:
- Install via package manager (apt-get, brew, etc.)
- Or specify path in `engine_interface.py`

**Low Classification Accuracy**:
- Increase `calibration.n_trials_per_target` for more training data
- Check EEG signal quality and electrode impedances
- Try `classifier: "xDAWN_LDA"` instead of `"LDA"`
- Ensure proper focus during calibration

## Project Context

Built for assistive technology research, specifically enabling locked-in syndrome patients to play chess. The P300 speller paradigm is well-established in BCI literature (Farwell & Donchin, 1988) and adapted here for chess move selection.

Key scientific references:
- xDAWN algorithm for P300 enhancement (Rivet et al., 2009)
- Riemannian geometry methods for BCI (Congedo et al., 2011)
- Standard P300 speller paradigm with row/column flashing

The system prioritizes:
- Reliability and robustness for clinical use
- Adaptive calibration for individual differences
- Clear visual feedback for users
- Accessibility for motor-impaired users
