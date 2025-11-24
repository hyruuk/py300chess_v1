# Running Without Installing Dependencies

## Problem

When you run `python src/main.py` without installing dependencies, you'll get import errors like:
```
ModuleNotFoundError: No module named 'pygame'
```

## Solution: Install Dependencies

The application requires several Python packages to run. Install them using:

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt

# Install Stockfish chess engine
sudo apt-get install stockfish  # Ubuntu/Debian
# OR
brew install stockfish  # macOS
```

## What You Can Do WITHOUT Dependencies

### 1. Verify Code Syntax
```bash
python tests/test_syntax.py
```
This checks that all Python files are syntactically correct without importing them.

### 2. Read the Documentation
- `README.md` - Main documentation
- `DEMO_MODE.md` - Keyboard controls guide
- `QUICKSTART.md` - Installation guide
- `plan.md` - Detailed implementation plan

### 3. Inspect the Code
Browse the source code in `src/` to understand the implementation.

## Minimal Installation (Just to Try Demo Mode)

If you only want to try the keyboard demo mode, you can install just the essential packages:

```bash
pip install pygame python-chess
sudo apt-get install stockfish
```

This will let you play chess with keyboard controls (no EEG needed).

## Full Installation

For the complete P300 BCI functionality:

```bash
pip install -r requirements.txt
```

This installs:
- `pygame` - GUI
- `python-chess` - Chess logic
- `stockfish` - Chess engine wrapper
- `numpy`, `scipy` - Numerical computing
- `scikit-learn` - Machine learning
- `mne` - EEG processing
- `pylsl` - Lab Streaming Layer (EEG acquisition)
- Other dependencies

## Quick Start After Installation

```bash
# Run the application
python src/main.py

# Press G to start playing!
# Use arrow keys and spacebar
```

## Troubleshooting

### "No module named X"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### "Stockfish not found"
**Solution**: Install Stockfish chess engine (see above)

### Virtual environment not activated
**Solution**:
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Permission errors on Linux
**Solution**: Use `pip install --user -r requirements.txt`

## Why Dependencies Are Required

| Package | Purpose | Required For |
|---------|---------|-------------|
| pygame | GUI rendering | All modes |
| python-chess | Chess rules | All modes |
| stockfish | Chess engine | Opponent play |
| numpy | Array operations | P300 mode |
| scipy | Signal processing | P300 mode |
| scikit-learn | Classification | P300 mode |
| mne | EEG analysis | P300 mode |
| pylsl | EEG acquisition | P300 mode |

Demo mode only needs: `pygame`, `python-chess`, `stockfish`

P300 mode needs: All of the above
