# Install and Run - Complete Guide

## Current Status

✅ **Code is ready** - All syntax errors fixed
⚠️ **Dependencies needed** - Must install packages to run

## Quick Install & Run

### Step 1: Install Dependencies

```bash
# Make sure you're in the project directory
cd /home/hyruuk/GitHub/kairos/p300chess

# Install all required packages
pip install -r requirements.txt
```

### Step 2: Install Stockfish Chess Engine

**Ubuntu/Debian:**
```bash
sudo apt-get install stockfish
```

**macOS:**
```bash
brew install stockfish
```

**Or check if already installed:**
```bash
which stockfish
```

### Step 3: Run the Application

```bash
python src/main.py
```

### Step 4: Play Chess!

Once the window opens:
1. Press **G** to start the game
2. Use **Arrow Keys** to move the cursor
3. Press **SPACE** to select piece and confirm move
4. Press **ESC** to cancel or return to menu

## Minimal Install (Just Demo Mode)

If you only want to test keyboard chess (no P300/EEG):

```bash
# Install only essential packages
pip install pygame python-chess PyYAML

# Install Stockfish
sudo apt-get install stockfish

# Run the app
python src/main.py
```

## Troubleshooting

### Error: "No module named 'numpy'"
**Cause**: Dependencies not installed
**Fix**: Run `pip install -r requirements.txt`

### Error: "No module named 'pygame'"
**Cause**: Pygame not installed
**Fix**: Run `pip install pygame`

### Error: "Stockfish not found"
**Cause**: Chess engine not installed
**Fix**: Run `sudo apt-get install stockfish` (or `brew install stockfish` on Mac)

### Error: "Permission denied"
**Fix**: Use virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### pygame display issues
**Cause**: No display available or Wayland issues
**Fix**: Set environment variable:
```bash
SDL_VIDEODRIVER=x11 python src/main.py
```

## What Gets Installed

### Essential (for demo mode):
- `pygame` - GUI framework
- `python-chess` - Chess rules engine
- `PyYAML` - Configuration file parsing

### Full (for P300 mode):
- `numpy` - Array operations
- `scipy` - Signal processing
- `scikit-learn` - Machine learning
- `mne` - EEG data analysis
- `pylsl` - EEG stream acquisition
- `pandas` - Data management
- `h5py` - Data storage

### External:
- `stockfish` - Chess AI opponent

## Installation Check

After installing, verify with:

```bash
# Check Python packages
pip list | grep -E 'pygame|chess|numpy|scipy|sklearn|mne|pylsl'

# Check Stockfish
which stockfish
stockfish --version
```

## Expected Output When Running

```
Initializing P300 Chess BCI Application...
Components initialized successfully!
============================================================
P300 Chess BCI Application
============================================================

Main Menu
```

Then a window opens showing:
- Title: "P300 Chess BCI"
- Mode indicator: "MODE: KEYBOARD DEMO" (in green)
- Instructions: "Press G to start Game"

## File Structure Reminder

After fixes applied:
- ✅ Package renamed: `chess/` → `game/`
- ✅ Import added: `Tuple` in feature_extraction.py
- ✅ All syntax validated
- ✅ Code ready to run

## Next Steps After Installation

1. **Test demo mode**: Press G to play chess with keyboard
2. **Adjust settings**: Edit `config/settings.yaml`
3. **Try different difficulty**: Change `engine_skill_level` (0-20)
4. **Read controls**: See `DEMO_MODE.md` for detailed instructions
5. **Set up EEG** (optional): For P300 mode, connect EEG hardware

## Need Help?

- **Installation issues**: See `RUN_WITHOUT_DEPENDENCIES.md`
- **Keyboard controls**: See `DEMO_MODE.md`
- **P300 mode**: See `README.md`
- **Code details**: See `plan.md`

## Summary

```bash
# Complete installation and run (3 commands):
pip install -r requirements.txt
sudo apt-get install stockfish  # or brew install stockfish
python src/main.py
```

That's it! You're ready to play chess with brain signals (or keyboard)! 🎮🧠♟️
