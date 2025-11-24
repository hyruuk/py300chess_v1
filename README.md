# P300 Chess BCI

A brain-computer interface (BCI) application that enables locked-in syndrome patients to play chess using P300 event-related potentials detected from EEG signals. The system uses a P300-speller paradigm where rows and columns of possible chess moves flash sequentially, and the desired move is selected based on which flash elicits the strongest P300 response.

## Features

- **P300-based Move Selection**: Uses row/column flashing paradigm similar to P300 speller
- **Adaptive Calibration**: Learns optimal parameters for each participant
- **Chess Engine Integration**: Play against Stockfish with adjustable difficulty
- **Real-time EEG Processing**: Filters, epochs, and classifies EEG data in real-time
- **Multiple Classification Methods**: LDA, xDAWN+LDA, and more
- **Accessible Design**: Built specifically for locked-in syndrome patients
- **Keyboard Demo Mode**: Test the system without EEG hardware! 🎮

## Quick Start (Demo Mode)

Want to try it out without EEG hardware? We've got you covered!

```bash
# 1. Install dependencies (see Installation section below)
pip install -r requirements.txt
sudo apt-get install stockfish  # or brew install stockfish on Mac

# 2. Run the application (demo mode is enabled by default)
python src/main.py

# 3. Press G to start playing!
# Use arrow keys to navigate, SPACE to select moves
```

Demo mode is enabled by default in `config/settings.yaml`. See [DEMO_MODE.md](DEMO_MODE.md) for detailed instructions.

## System Requirements

### Hardware
- EEG amplifier compatible with Lab Streaming Layer (LSL)
- Minimum 8 EEG channels (Fz, Cz, Pz, Oz, P3, P4, C3, C4 recommended)
- Computer with Python 3.8 or higher

### Software Dependencies
See `requirements.txt` for full list. Key dependencies:
- pylsl (EEG acquisition)
- numpy, scipy (signal processing)
- scikit-learn (machine learning)
- mne (EEG analysis)
- python-chess (chess logic)
- stockfish (chess engine)
- pygame (GUI)

## Installation

1. **Clone the repository**:
```bash
cd /home/hyruuk/GitHub/kairos/p300chess
```

2. **Create and activate virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Stockfish chess engine**:

Ubuntu/Debian:
```bash
sudo apt-get install stockfish
```

MacOS:
```bash
brew install stockfish
```

Windows:
Download from [stockfishchess.org](https://stockfishchess.org/download/)

5. **Set up LSL stream**:
- Configure your EEG amplifier to stream data via LSL
- Ensure stream name matches configuration (default: "EEG")

## Configuration

Edit `config/settings.yaml` to customize:

- **EEG settings**: Stream name, sampling rate, channels
- **Signal processing**: Filter parameters, epoch windows
- **P300 detection**: Classifier type, target channels
- **Calibration**: Number of trials, flash timing
- **Chess game**: Engine difficulty, flash parameters
- **GUI**: Screen size, colors, board size

## Usage

### Running the Application

```bash
python src/main.py
```

Or with custom config:
```bash
python src/main.py --config path/to/settings.yaml
```

### Application Flow

1. **Main Menu**
   - Press `C` for Calibration
   - Press `G` for Game (requires calibration first)
   - Press `ESC` to quit

2. **Calibration**
   - Visual stimuli (rows/columns) flash on screen
   - Focus attention on the target stimulus
   - System collects EEG data and trains classifier
   - Typically takes 10-15 minutes

3. **Playing Chess**
   - **Move Selection** (2-step process):
     1. Rows and columns flash to select origin square
     2. Rows and columns flash again to select destination square
   - System detects P300 response to determine selected move
   - Engine responds with its move
   - Continue until game ends

### Keyboard Controls

- `ESC`: Quit application / Return to menu
- `C`: Start calibration
- `G`: Start game
- `M`: Return to menu (during game)

## Project Structure

```
p300chess/
├── src/
│   ├── acquisition/       # EEG data acquisition
│   │   ├── lsl_client.py     # LSL stream connection
│   │   └── data_buffer.py    # Circular buffer for EEG data
│   ├── processing/        # Signal processing
│   │   ├── preprocessing.py   # Filtering, epoching
│   │   └── feature_extraction.py  # Feature extraction
│   ├── classification/    # P300 detection
│   │   ├── calibration.py     # Calibration procedure
│   │   └── p300_detector.py   # P300 classifier
│   ├── chess/            # Chess game logic
│   │   ├── game_manager.py    # Game state management
│   │   ├── move_selector.py   # P300-based move selection
│   │   └── engine_interface.py # Stockfish interface
│   ├── gui/              # User interface
│   │   ├── main_window.py     # Main application window
│   │   ├── chess_board.py     # Board visualization
│   │   └── flash_controller.py # Flash timing control
│   └── main.py           # Application entry point
├── config/
│   └── settings.yaml     # Configuration file
├── data/                 # Session data storage
├── models/               # Trained classifier models
├── logs/                 # Application logs
└── tests/                # Unit tests
```

## How It Works

### P300 Event-Related Potential

The P300 is a positive voltage deflection in EEG that occurs ~300ms after a rare or task-relevant stimulus. When a user focuses attention on a target stimulus (e.g., a flashing row containing their desired move), it elicits a stronger P300 response than non-target stimuli.

### Move Selection Paradigm

1. **Row/Column Flashing**:
   - 8 rows (ranks 1-8) and 8 columns (files a-h) flash in random order
   - Each row/column flashes multiple times (typically 5 repetitions)
   - User focuses attention on row/column containing desired square

2. **P300 Detection**:
   - System extracts EEG epochs time-locked to each flash
   - Applies preprocessing (filtering, baseline correction)
   - Classifier predicts which flashes elicited P300 responses
   - Intersection of selected row and column determines the square

3. **Two-Step Selection**:
   - Step 1: Select origin square (where piece is)
   - Step 2: Select destination square (where to move piece)
   - System validates and executes legal move

### Signal Processing Pipeline

```
EEG Data → Bandpass Filter (0.1-20 Hz) → Notch Filter (50/60 Hz) →
Epoch Extraction (-0.2 to 0.8s) → Baseline Correction →
Feature Extraction → Classification → Move Selection
```

### Classification Methods

1. **LDA** (Linear Discriminant Analysis):
   - Simple, fast, robust baseline method
   - Works well with limited training data

2. **xDAWN + LDA**:
   - xDAWN spatial filtering enhances P300 signal
   - Improves signal-to-noise ratio
   - Better accuracy than LDA alone

## Calibration

The calibration phase is crucial for good performance:

- **Duration**: ~10-15 minutes
- **Trials**: 15 repetitions per target (configurable)
- **Process**: User focuses on indicated target while stimuli flash
- **Output**: Trained classifier optimized for the participant

### Tips for Good Calibration

1. Minimize eye movements and muscle artifacts
2. Maintain focus on target stimulus
3. Stay relaxed but attentive
4. Take breaks if fatigued
5. Ensure good EEG signal quality

## Performance

Typical performance metrics:

- **Classification Accuracy**: 70-90% (depends on participant and calibration)
- **Move Selection Time**: ~20-30 seconds per move
- **Flash Repetitions**: 5 cycles (adjustable for speed vs. accuracy trade-off)

## Extending the System

### Adding New Chess Engines

Edit `src/chess/engine_interface.py` to support additional UCI-compatible engines.

### Custom Classification Methods

Implement new classifiers in `src/classification/p300_detector.py`. The system supports any scikit-learn compatible classifier.

### Alternative Selection Paradigms

Modify `src/chess/move_selector.py` to implement different selection strategies:
- Piece-based selection
- Direct move selection
- Checkerboard pattern flashing

## Troubleshooting

### Cannot Connect to LSL Stream
- Verify EEG amplifier is streaming via LSL
- Check stream name in config matches actual stream
- Use `pylsl.resolve_streams()` to list available streams

### Low Classification Accuracy
- Increase number of calibration trials
- Check EEG signal quality
- Adjust preprocessing parameters
- Try different classifier (xDAWN+LDA vs LDA)
- Ensure user maintains focus during calibration

### Stockfish Not Found
- Install Stockfish: `sudo apt-get install stockfish`
- Or specify path in `src/chess/engine_interface.py`

### GUI Performance Issues
- Reduce screen resolution in config
- Lower frame rate
- Disable fullscreen mode

## References

### Scientific Papers
- Farwell & Donchin (1988): "Talking off the top of your head" - Original P300 speller
- Rivet et al. (2009): "xDAWN Algorithm to Enhance Evoked Potentials"
- Congedo et al. (2011): "Riemannian Geometry for EEG-based BCI"

### Libraries
- [MNE-Python](https://mne.tools/): EEG analysis
- [python-chess](https://python-chess.readthedocs.io/): Chess logic
- [LSL](https://labstreaminglayer.readthedocs.io/): Data streaming
- [Stockfish](https://stockfishchess.org/): Chess engine

## Contributing

Contributions are welcome! Areas for improvement:

- Enhanced calibration procedures
- Additional classification methods
- Online classifier adaptation
- Advanced move selection paradigms
- Multi-modal BCI (P300 + motor imagery)
- Mobile/tablet support

## License

This project is intended for research and assistive technology purposes.

## Acknowledgments

Built for locked-in syndrome patients to provide cognitive stimulation and social engagement through chess.

## Contact

For questions, issues, or contributions, please open an issue on the project repository.
