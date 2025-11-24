# Quick Start Guide

## Demo Mode (No EEG Required!) 🎮

The fastest way to try the system:

```bash
# Run the application (demo mode enabled by default)
python src/main.py

# Press G to start playing chess with keyboard controls!
# Arrow keys to move, SPACE to select
```

See [DEMO_MODE.md](DEMO_MODE.md) for complete keyboard controls.

---

## Installation

### 1. Install Dependencies

First, activate your virtual environment and install required packages:

```bash
# Create virtual environment (if not already created)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Install Stockfish Chess Engine

**Ubuntu/Debian:**
```bash
sudo apt-get install stockfish
```

**MacOS:**
```bash
brew install stockfish
```

**Windows:**
Download from https://stockfishchess.org/download/

### 3. Verify Installation

**Option A: Quick syntax check (no dependencies needed)**
```bash
python tests/test_syntax.py
```

**Option B: Full tests (requires dependencies)**
```bash
python tests/test_basic.py
```

Expected output for syntax test:
```
Testing Python syntax...
✓ All 20 files checked
✅ All files have valid syntax!
```

## Running the Application

### Without EEG (GUI Demo)

To test the GUI and chess interface without EEG:

```bash
python src/main.py
```

This will:
- Launch the main window
- Show the menu (press C for calibration, G for game)
- Display the chess board
- Note: Full functionality requires EEG stream connection

### With EEG Hardware

1. **Set up LSL stream:**
   - Start your EEG amplifier's LSL streaming application
   - Verify stream is broadcasting (check stream name)

2. **Update configuration:**
   Edit `config/settings.yaml` to match your setup:
   ```yaml
   eeg:
     stream_name: "YourStreamName"  # Match your LSL stream
     sampling_rate: 250  # Match your hardware
     channels: ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'C3', 'C4']
   ```

3. **Run application:**
   ```bash
   python src/main.py
   ```

4. **Workflow:**
   - Press `C` to start calibration
   - Complete calibration session (~10-15 min)
   - Press `G` to start playing chess
   - Use P300 responses to select moves

## Testing Individual Components

### Test Chess Logic

```python
from src.chess.game_manager import ChessGameManager

game = ChessGameManager()
print(f"Legal moves: {len(game.get_legal_moves())}")
game.make_move_uci("e2e4")
print(f"Board: {game.get_fen()}")
```

### Test P300 Detector

```python
import numpy as np
from src.classification.p300_detector import P300Detector

# Create synthetic training data
X = np.random.randn(100, 200, 8)  # 100 epochs, 200 samples, 8 channels
y = np.random.randint(0, 2, 100)  # Binary labels

# Train detector
detector = P300Detector(method='LDA')
metrics = detector.train(X, y)
print(f"Accuracy: {metrics['cv_accuracy']:.3f}")
```

### Test Signal Processing

```python
import numpy as np
from src.processing.preprocessing import EEGPreprocessor

preprocessor = EEGPreprocessor(
    sampling_rate=250,
    bandpass=(0.1, 20),
    notch_freq=50
)

# Filter some data
data = np.random.randn(1000, 8)
filtered = preprocessor.filter(data)
print(f"Filtered data shape: {filtered.shape}")
```

## Configuration

Key settings in `config/settings.yaml`:

### For Testing/Development
- `gui.fullscreen: false` - Use windowed mode
- `calibration.n_trials_per_stimulus: 5` - Reduce trials for faster testing
- `chess.n_repetitions: 3` - Fewer flashes for faster moves

### For Production/Real Use
- `gui.fullscreen: true` - Full screen for better focus
- `calibration.n_trials_per_stimulus: 15` - More trials for better accuracy
- `chess.n_repetitions: 5` - More repetitions for reliable detection

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'X'
```
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Stockfish Not Found
```
FileNotFoundError: Stockfish not found
```
**Solution:** Install Stockfish or specify path in `src/chess/engine_interface.py`

### Cannot Connect to LSL
```
No stream found with name: EEG
```
**Solution:**
- Verify EEG amplifier is streaming
- Check stream name matches config
- Test with: `python -c "import pylsl; print(pylsl.resolve_streams())"`

### Low Frame Rate
**Solution:** Reduce resolution in config or disable fullscreen

## Next Steps

1. **Test Without EEG:**
   - Run basic tests
   - Explore GUI
   - Test chess engine

2. **Set Up EEG:**
   - Configure LSL stream
   - Verify signal quality
   - Test data acquisition

3. **Run Calibration:**
   - Complete full calibration
   - Save trained model
   - Evaluate accuracy

4. **Play Chess:**
   - Start game mode
   - Practice move selection
   - Adjust parameters as needed

5. **Optimize:**
   - Tune classifier parameters
   - Adjust flash timing
   - Improve accuracy

## Support

- See `README.md` for detailed documentation
- See `plan.md` for implementation details
- Run `python src/main.py --help` for command-line options
