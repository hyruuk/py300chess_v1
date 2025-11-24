# P300-Based BCI Chess Application - Development Plan

## Project Overview

A Python application enabling locked-in syndrome patients to play chess using brain-computer interface (BCI) technology. The system uses P300 event-related potentials (ERPs) detected from EEG signals streamed via Lab Streaming Layer (LSL). Move selection uses a P300-speller paradigm where rows and columns of possible moves flash sequentially, and the system detects which move elicits the strongest P300 response.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     P300 Chess BCI System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ EEG Hardware │───>│  LSL Stream  │───>│ Data Buffer  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │           │
│                                                   v           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Signal Processing Pipeline                 │   │
│  │  • Filtering (bandpass, notch)                       │   │
│  │  • Epoching (time-locked to flash events)            │   │
│  │  • Artifact rejection                                 │   │
│  │  • Feature extraction                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                   │           │
│                                                   v           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              P300 Detection System                    │   │
│  │  • Calibration mode (parameter learning)             │   │
│  │  • Classification (LDA, xDAWN+Riemannian, etc.)      │   │
│  │  • Confidence scoring                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                   │           │
│                                                   v           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Chess Game Interface & Logic                  │   │
│  │  • Board state management                             │   │
│  │  • Move validation                                    │   │
│  │  • Visual presentation (row/col highlighting)         │   │
│  │  • Flash paradigm controller                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                   │           │
│                                                   v           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Chess Engine Interface                   │   │
│  │  • Stockfish integration                              │   │
│  │  • Difficulty/ELO settings                            │   │
│  │  • Multiple engine support (future)                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Required Dependencies

### Core Libraries
- **pylsl**: Lab Streaming Layer Python interface for EEG data acquisition
- **numpy**: Numerical operations and array handling
- **scipy**: Signal processing (filtering, etc.)
- **scikit-learn**: Machine learning classifiers (LDA, SVM, etc.)
- **mne**: EEG/MEG data processing and analysis
- **pyriemann**: Riemannian geometry-based classification (optional, for advanced methods)

### Chess Libraries
- **python-chess**: Chess move generation, validation, and board representation
- **stockfish**: Python wrapper for Stockfish chess engine

### GUI/Visualization
- **pygame**: Game interface and visual stimulus presentation
- **psychopy**: Alternative for precise stimulus timing (optional)
- **matplotlib**: Data visualization for calibration/debugging

### Data Management
- **pandas**: Data organization and export
- **h5py** or **pickle**: Session data storage

## Detailed Implementation Steps

### Phase 1: Project Setup and Infrastructure

#### Step 1.1: Environment Setup
```bash
# Create project structure
mkdir -p p300chess/{src,config,data,models,logs,tests}
mkdir -p p300chess/src/{acquisition,processing,classification,chess,gui}

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pylsl numpy scipy scikit-learn mne python-chess stockfish pygame pandas h5py
pip install pyriemann  # Optional, for advanced methods
```

#### Step 1.2: Configuration System
Create `config/settings.yaml`:
```yaml
# EEG Acquisition
eeg:
  stream_name: "EEG"  # LSL stream name to connect to
  sampling_rate: 250  # Expected sampling rate (Hz)
  channels: ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'C3', 'C4']  # Target channels
  buffer_length: 10  # Seconds of data to buffer

# Signal Processing
processing:
  bandpass_low: 0.1  # Hz
  bandpass_high: 20  # Hz
  notch_freq: 50  # Hz (60 for North America)
  epoch_tmin: -0.2  # Seconds before stimulus
  epoch_tmax: 0.8  # Seconds after stimulus
  baseline: [-0.2, 0]  # Baseline correction window

# P300 Detection
p300:
  target_channels: ['Pz', 'Cz', 'P3', 'P4']  # Primary P300 channels
  classifier: "xDAWN_LDA"  # Options: LDA, xDAWN_LDA, Riemannian, ensemble
  n_xdawn_components: 4
  cross_validation_folds: 5

# Calibration
calibration:
  n_trials_per_stimulus: 15  # Number of repetitions per target
  n_nontarget_per_target: 5  # Non-target stimuli per target
  flash_duration: 0.1  # Seconds
  isi: 0.3  # Inter-stimulus interval (seconds)
  feedback_enabled: true

# Chess Game
chess:
  flash_duration: 0.15  # Seconds
  isi: 0.25  # Inter-stimulus interval
  n_repetitions: 5  # Flash cycles per selection
  engine: "stockfish"
  engine_skill_level: 5  # 0-20 for Stockfish
  engine_depth: 10  # Search depth
  move_confirmation: true  # Require confirmation before executing

# GUI
gui:
  screen_width: 1920
  screen_height: 1080
  fullscreen: true
  background_color: [40, 40, 40]
  flash_color: [255, 255, 100]
  board_size: 640  # Pixels
  font_size: 24
```

#### Step 1.3: Project Structure
```
p300chess/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── lsl_client.py       # LSL stream acquisition
│   │   └── data_buffer.py      # Circular buffer for incoming data
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Filtering, epoching
│   │   ├── artifact_rejection.py
│   │   └── feature_extraction.py
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── calibration.py      # Calibration procedure
│   │   ├── p300_detector.py    # P300 classification
│   │   └── models.py           # Classifier implementations
│   ├── chess/
│   │   ├── __init__.py
│   │   ├── game_manager.py     # Chess game state
│   │   ├── move_selector.py    # P300-based move selection
│   │   └── engine_interface.py # Chess engine communication
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py      # Main application window
│       ├── calibration_screen.py
│       ├── chess_board.py      # Board rendering
│       └── flash_controller.py # Stimulus presentation
├── config/
│   └── settings.yaml
├── data/
│   ├── calibration/
│   └── sessions/
├── models/
│   └── classifiers/
├── logs/
├── tests/
└── requirements.txt
```

### Phase 2: EEG Data Acquisition

#### Step 2.1: LSL Client Implementation (`src/acquisition/lsl_client.py`)
```python
"""
LSL stream client for acquiring EEG data in real-time.
Connects to an LSL stream and provides data to downstream components.
"""

import pylsl
import numpy as np
import threading
import time
from typing import Optional, Callable, List

class LSLClient:
    def __init__(self, stream_name: str, expected_channels: List[str],
                 sampling_rate: int):
        """
        Initialize LSL client.

        Args:
            stream_name: Name of the LSL stream to connect to
            expected_channels: List of expected channel names
            sampling_rate: Expected sampling rate in Hz
        """
        self.stream_name = stream_name
        self.expected_channels = expected_channels
        self.sampling_rate = sampling_rate
        self.inlet = None
        self.running = False
        self.thread = None

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to LSL stream.

        Returns:
            True if connection successful, False otherwise
        """
        # Resolve stream
        streams = pylsl.resolve_byprop('name', self.stream_name,
                                       timeout=timeout)
        if not streams:
            print(f"No stream found with name: {self.stream_name}")
            return False

        # Create inlet
        self.inlet = pylsl.StreamInlet(streams[0], max_buflen=360,
                                       processing_flags=pylsl.proc_clocksync)

        # Verify stream info
        info = self.inlet.info()
        n_channels = info.channel_count()
        actual_sr = info.nominal_srate()

        print(f"Connected to stream: {self.stream_name}")
        print(f"Channels: {n_channels}, Sampling rate: {actual_sr} Hz")

        return True

    def start_acquisition(self, callback: Callable):
        """
        Start acquiring data in a background thread.

        Args:
            callback: Function to call with (sample, timestamp) for each sample
        """
        self.running = True
        self.thread = threading.Thread(target=self._acquisition_loop,
                                       args=(callback,))
        self.thread.start()

    def _acquisition_loop(self, callback: Callable):
        """Background acquisition loop."""
        while self.running:
            sample, timestamp = self.inlet.pull_sample(timeout=0.0)
            if sample:
                callback(np.array(sample), timestamp)
            else:
                time.sleep(0.001)

    def stop_acquisition(self):
        """Stop data acquisition."""
        self.running = False
        if self.thread:
            self.thread.join()
```

#### Step 2.2: Data Buffer Implementation (`src/acquisition/data_buffer.py`)
```python
"""
Circular buffer for storing incoming EEG data with event markers.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, List

class DataBuffer:
    def __init__(self, n_channels: int, sampling_rate: int,
                 buffer_length: float = 10.0):
        """
        Initialize circular buffer.

        Args:
            n_channels: Number of EEG channels
            sampling_rate: Sampling rate in Hz
            buffer_length: Buffer length in seconds
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.buffer_size = int(buffer_length * sampling_rate)

        # Data storage
        self.data = np.zeros((self.buffer_size, n_channels))
        self.timestamps = np.zeros(self.buffer_size)
        self.write_index = 0
        self.samples_written = 0

        # Event markers
        self.events = []  # List of (timestamp, event_code, event_info)

    def add_sample(self, sample: np.ndarray, timestamp: float):
        """Add a new sample to the buffer."""
        self.data[self.write_index] = sample
        self.timestamps[self.write_index] = timestamp
        self.write_index = (self.write_index + 1) % self.buffer_size
        self.samples_written += 1

    def add_event(self, timestamp: float, event_code: int,
                  event_info: dict = None):
        """Add an event marker."""
        self.events.append({
            'timestamp': timestamp,
            'code': event_code,
            'info': event_info or {}
        })

    def get_epoch(self, event_timestamp: float, tmin: float, tmax: float) -> \
                  Optional[np.ndarray]:
        """
        Extract an epoch around an event.

        Args:
            event_timestamp: Event timestamp
            tmin: Start time relative to event (seconds, negative)
            tmax: End time relative to event (seconds, positive)

        Returns:
            Epoch data as (n_samples, n_channels) array, or None if unavailable
        """
        # Find samples in time window
        start_time = event_timestamp + tmin
        end_time = event_timestamp + tmax

        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        epoch_data = self.data[mask]

        if len(epoch_data) == 0:
            return None

        return epoch_data
```

### Phase 3: Signal Processing Pipeline

#### Step 3.1: Preprocessing (`src/processing/preprocessing.py`)
```python
"""
Signal preprocessing: filtering, epoching, baseline correction.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional

class EEGPreprocessor:
    def __init__(self, sampling_rate: int, bandpass: Tuple[float, float],
                 notch_freq: Optional[float] = None):
        """
        Initialize preprocessor.

        Args:
            sampling_rate: Sampling rate in Hz
            bandpass: (low, high) cutoff frequencies in Hz
            notch_freq: Notch filter frequency (50 or 60 Hz)
        """
        self.sampling_rate = sampling_rate
        self.bandpass = bandpass
        self.notch_freq = notch_freq

        # Design filters
        self._design_filters()

    def _design_filters(self):
        """Design bandpass and notch filters."""
        nyquist = self.sampling_rate / 2

        # Bandpass filter (Butterworth, 4th order)
        low = self.bandpass[0] / nyquist
        high = self.bandpass[1] / nyquist
        self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')

        # Notch filter (if specified)
        if self.notch_freq:
            w0 = self.notch_freq / nyquist
            Q = 30  # Quality factor
            self.notch_b, self.notch_a = signal.iirnotch(w0, Q)

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filtering to data.

        Args:
            data: Input data (n_samples, n_channels)

        Returns:
            Filtered data
        """
        filtered = np.zeros_like(data)

        for ch in range(data.shape[1]):
            # Bandpass filter
            filtered[:, ch] = signal.filtfilt(self.bp_b, self.bp_a,
                                              data[:, ch])

            # Notch filter
            if self.notch_freq:
                filtered[:, ch] = signal.filtfilt(self.notch_b, self.notch_a,
                                                  filtered[:, ch])

        return filtered

    def baseline_correct(self, epoch: np.ndarray, baseline_window: Tuple[int, int]) -> \
                         np.ndarray:
        """
        Apply baseline correction.

        Args:
            epoch: Epoch data (n_samples, n_channels)
            baseline_window: (start, end) sample indices for baseline

        Returns:
            Baseline-corrected epoch
        """
        baseline = epoch[baseline_window[0]:baseline_window[1], :].mean(axis=0)
        return epoch - baseline
```

#### Step 3.2: Feature Extraction (`src/processing/feature_extraction.py`)
```python
"""
Feature extraction for P300 classification.
"""

import numpy as np
from typing import List

class P300FeatureExtractor:
    def __init__(self, target_channels: List[int], sampling_rate: int):
        """
        Initialize feature extractor.

        Args:
            target_channels: Indices of channels to use for features
            sampling_rate: Sampling rate in Hz
        """
        self.target_channels = target_channels
        self.sampling_rate = sampling_rate

    def extract_temporal_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract temporal features (downsampled epoch).

        Args:
            epoch: Single epoch (n_samples, n_channels)

        Returns:
            Feature vector
        """
        # Select target channels
        epoch_subset = epoch[:, self.target_channels]

        # Downsample (e.g., every 4th sample for 250Hz -> ~60Hz)
        downsampled = epoch_subset[::4, :]

        # Flatten to feature vector
        features = downsampled.flatten()

        return features

    def extract_spatial_features(self, epoch: np.ndarray,
                                  spatial_filters: np.ndarray) -> np.ndarray:
        """
        Extract spatial features using learned filters (e.g., xDAWN).

        Args:
            epoch: Single epoch (n_samples, n_channels)
            spatial_filters: Spatial filter matrix (n_channels, n_components)

        Returns:
            Spatially filtered epoch
        """
        # Apply spatial filters
        filtered = epoch @ spatial_filters

        # Downsample and flatten
        downsampled = filtered[::4, :]
        features = downsampled.flatten()

        return features
```

### Phase 4: Calibration System

#### Step 4.1: Calibration Procedure (`src/classification/calibration.py`)
```python
"""
Calibration procedure for P300 detection.
Presents visual stimuli and collects labeled training data.
"""

import numpy as np
from typing import List, Tuple
import random

class CalibrationSession:
    def __init__(self, n_targets: int, n_trials_per_target: int,
                 n_nontargets_per_target: int, flash_duration: float,
                 isi: float):
        """
        Initialize calibration session.

        Args:
            n_targets: Number of unique target stimuli
            n_trials_per_target: Repetitions of each target
            n_nontargets_per_target: Non-targets to show per target
            flash_duration: Flash duration in seconds
            isi: Inter-stimulus interval in seconds
        """
        self.n_targets = n_targets
        self.n_trials_per_target = n_trials_per_target
        self.n_nontargets_per_target = n_nontargets_per_target
        self.flash_duration = flash_duration
        self.isi = isi

        # Generate trial sequence
        self.trials = self._generate_trials()
        self.current_trial = 0

        # Data collection
        self.epochs = []
        self.labels = []

    def _generate_trials(self) -> List[dict]:
        """Generate randomized trial sequence."""
        trials = []

        for target_id in range(self.n_targets):
            for trial in range(self.n_trials_per_target):
                # Create a sequence with target and non-targets
                sequence = [{'stimulus_id': target_id, 'is_target': True}]

                # Add non-targets (other stimuli)
                nontarget_ids = [i for i in range(self.n_targets) if i != target_id]
                selected_nontargets = random.sample(nontarget_ids,
                                                   self.n_nontargets_per_target)

                for nt_id in selected_nontargets:
                    sequence.append({'stimulus_id': nt_id, 'is_target': False})

                # Randomize sequence
                random.shuffle(sequence)

                trials.append({
                    'target_id': target_id,
                    'sequence': sequence
                })

        # Randomize trial order
        random.shuffle(trials)
        return trials

    def get_current_trial(self) -> dict:
        """Get current trial information."""
        if self.current_trial < len(self.trials):
            return self.trials[self.current_trial]
        return None

    def next_trial(self):
        """Advance to next trial."""
        self.current_trial += 1

    def add_epoch(self, epoch: np.ndarray, is_target: bool):
        """Add a labeled epoch to training data."""
        self.epochs.append(epoch)
        self.labels.append(1 if is_target else 0)

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get collected training data."""
        X = np.array(self.epochs)
        y = np.array(self.labels)
        return X, y
```

#### Step 4.2: P300 Classifier (`src/classification/p300_detector.py`)
```python
"""
P300 detection classifier with multiple algorithm options.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional

class P300Detector:
    def __init__(self, method: str = 'LDA'):
        """
        Initialize P300 detector.

        Args:
            method: Classification method ('LDA', 'xDAWN_LDA', 'Riemannian')
        """
        self.method = method
        self.classifier = None
        self.spatial_filters = None
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the classifier.

        Args:
            X: Training epochs (n_epochs, n_samples, n_channels)
            y: Labels (0=non-target, 1=target)

        Returns:
            Dictionary with training metrics
        """
        # Flatten epochs to feature vectors
        n_epochs = X.shape[0]
        X_flat = X.reshape(n_epochs, -1)

        if self.method == 'LDA':
            self.classifier = LinearDiscriminantAnalysis(solver='lsqr',
                                                         shrinkage='auto')

        elif self.method == 'xDAWN_LDA':
            # Simplified xDAWN: compute spatial filters based on targets
            self.spatial_filters = self._compute_xdawn_filters(X, y,
                                                               n_components=4)
            # Apply spatial filtering
            X_filtered = self._apply_spatial_filters(X)
            X_flat = X_filtered.reshape(n_epochs, -1)

            self.classifier = LinearDiscriminantAnalysis(solver='lsqr',
                                                         shrinkage='auto')

        # Train classifier
        self.classifier.fit(X_flat, y)

        # Cross-validation score
        cv_scores = cross_val_score(self.classifier, X_flat, y, cv=5)

        self.is_trained = True

        return {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_targets': np.sum(y == 1),
            'n_nontargets': np.sum(y == 0)
        }

    def predict(self, epoch: np.ndarray) -> Tuple[int, float]:
        """
        Predict if epoch contains P300.

        Args:
            epoch: Single epoch (n_samples, n_channels)

        Returns:
            (prediction, confidence) where prediction is 0 or 1
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained")

        # Apply spatial filters if using xDAWN
        if self.spatial_filters is not None:
            epoch_filtered = epoch @ self.spatial_filters
            features = epoch_filtered.flatten()
        else:
            features = epoch.flatten()

        # Predict
        prediction = self.classifier.predict([features])[0]
        confidence = self.classifier.predict_proba([features])[0][1]

        return int(prediction), float(confidence)

    def _compute_xdawn_filters(self, X: np.ndarray, y: np.ndarray,
                               n_components: int = 4) -> np.ndarray:
        """
        Compute xDAWN spatial filters.
        (Simplified implementation - consider using pyriemann for full version)
        """
        # Average target and non-target epochs
        X_target = X[y == 1].mean(axis=0)  # (n_samples, n_channels)
        X_nontarget = X[y == 0].mean(axis=0)

        # Compute covariance matrices
        C_target = np.cov(X_target.T)
        C_total = np.cov(X.reshape(-1, X.shape[2]).T)

        # Generalized eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(
            np.linalg.inv(C_total) @ C_target
        )

        # Sort by eigenvalues and select top components
        idx = np.argsort(eigenvalues)[::-1]
        spatial_filters = eigenvectors[:, idx[:n_components]].real

        return spatial_filters

    def _apply_spatial_filters(self, X: np.ndarray) -> np.ndarray:
        """Apply spatial filters to epochs."""
        n_epochs = X.shape[0]
        X_filtered = np.zeros((n_epochs, X.shape[1],
                              self.spatial_filters.shape[1]))

        for i in range(n_epochs):
            X_filtered[i] = X[i] @ self.spatial_filters

        return X_filtered
```

### Phase 5: Chess Game Logic

#### Step 5.1: Game Manager (`src/chess/game_manager.py`)
```python
"""
Chess game state management using python-chess library.
"""

import chess
from typing import List, Optional, Tuple

class ChessGameManager:
    def __init__(self):
        """Initialize chess game."""
        self.board = chess.Board()
        self.move_history = []

    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves in current position."""
        return list(self.board.legal_moves)

    def get_legal_moves_grouped(self) -> dict:
        """
        Get legal moves grouped by origin square and destination.
        Useful for row/column flashing paradigm.

        Returns:
            Dictionary with move groupings
        """
        moves = self.get_legal_moves()

        # Group by origin square
        by_origin = {}
        for move in moves:
            origin = chess.square_name(move.from_square)
            if origin not in by_origin:
                by_origin[origin] = []
            by_origin[origin].append(move)

        # Group by destination square
        by_dest = {}
        for move in moves:
            dest = chess.square_name(move.to_square)
            if dest not in by_dest:
                by_dest[dest] = []
            by_dest[dest].append(move)

        return {
            'by_origin': by_origin,
            'by_destination': by_dest,
            'all_moves': moves
        }

    def make_move(self, move: chess.Move) -> bool:
        """
        Make a move if legal.

        Returns:
            True if move was made, False if illegal
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            return True
        return False

    def undo_move(self) -> Optional[chess.Move]:
        """Undo last move."""
        if self.board.move_stack:
            move = self.board.pop()
            self.move_history.pop()
            return move
        return None

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()

    def get_result(self) -> Optional[str]:
        """Get game result if game is over."""
        if self.is_game_over():
            return self.board.result()
        return None

    def get_fen(self) -> str:
        """Get current position in FEN notation."""
        return self.board.fen()
```

#### Step 5.2: Move Selection via P300 (`src/chess/move_selector.py`)
```python
"""
P300-based move selection using row/column flashing paradigm.
"""

import chess
import numpy as np
from typing import List, Dict, Tuple, Optional

class P300MoveSelector:
    def __init__(self, flash_duration: float, isi: float, n_repetitions: int):
        """
        Initialize move selector.

        Args:
            flash_duration: Flash duration in seconds
            isi: Inter-stimulus interval in seconds
            n_repetitions: Number of flash cycles for selection
        """
        self.flash_duration = flash_duration
        self.isi = isi
        self.n_repetitions = n_repetitions

    def create_flash_sequence(self, legal_moves: List[chess.Move]) -> List[dict]:
        """
        Create row/column flash sequence for moves.

        Strategy: Flash rows and columns of the chessboard. Detect which
        row and column elicit strongest P300 to determine selected square.

        Returns:
            List of flash events with metadata
        """
        # Create groups for rows (1-8) and columns (a-h)
        rows = list(range(8))
        cols = list(range(8))

        flash_sequence = []

        for rep in range(self.n_repetitions):
            # Randomize order for this repetition
            stimuli = []

            # Add row flashes
            for row in rows:
                stimuli.append({
                    'type': 'row',
                    'index': row,
                    'squares': [chess.square(col, row) for col in range(8)]
                })

            # Add column flashes
            for col in cols:
                stimuli.append({
                    'type': 'col',
                    'index': col,
                    'squares': [chess.square(col, row) for row in range(8)]
                })

            # Randomize stimuli for this repetition
            np.random.shuffle(stimuli)
            flash_sequence.extend(stimuli)

        return flash_sequence

    def determine_selected_square(self, flash_sequence: List[dict],
                                   p300_scores: List[float]) -> Optional[int]:
        """
        Determine which square was selected based on P300 scores.

        Args:
            flash_sequence: The flash sequence that was presented
            p300_scores: P300 detection scores for each flash

        Returns:
            Selected square index, or None if cannot determine
        """
        # Aggregate scores by row and column
        row_scores = np.zeros(8)
        col_scores = np.zeros(8)

        for flash, score in zip(flash_sequence, p300_scores):
            if flash['type'] == 'row':
                row_scores[flash['index']] += score
            elif flash['type'] == 'col':
                col_scores[flash['index']] += score

        # Find best row and column
        best_row = np.argmax(row_scores)
        best_col = np.argmax(col_scores)

        # Determine square
        selected_square = chess.square(best_col, best_row)

        return selected_square

    def select_move_two_step(self, legal_moves: List[chess.Move]) -> dict:
        """
        Two-step selection: first select origin square, then destination.

        Returns:
            Dictionary with flash sequences for both steps
        """
        # Step 1: Select origin square (from square)
        origin_squares = list(set([move.from_square for move in legal_moves]))

        # Step 2: Will be determined after step 1 completes
        # (filter legal moves by selected origin)

        return {
            'step': 'select_origin',
            'origin_squares': origin_squares
        }
```

#### Step 5.3: Chess Engine Interface (`src/chess/engine_interface.py`)
```python
"""
Interface to chess engines (Stockfish initially, extensible to others).
"""

import chess
import chess.engine
from typing import Optional

class ChessEngineInterface:
    def __init__(self, engine_path: str = '/usr/games/stockfish',
                 skill_level: int = 5, depth: int = 10):
        """
        Initialize chess engine.

        Args:
            engine_path: Path to engine executable
            skill_level: Skill level (0-20 for Stockfish)
            depth: Search depth
        """
        self.engine_path = engine_path
        self.skill_level = skill_level
        self.depth = depth
        self.engine = None

    def start(self):
        """Start the chess engine."""
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

        # Set skill level for Stockfish
        self.engine.configure({"Skill Level": self.skill_level})

    def stop(self):
        """Stop the chess engine."""
        if self.engine:
            self.engine.quit()

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get best move for current position.

        Args:
            board: Current board state

        Returns:
            Best move according to engine
        """
        if not self.engine:
            raise RuntimeError("Engine not started")

        result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        return result.move

    def analyze_position(self, board: chess.Board) -> dict:
        """
        Analyze position and return evaluation.

        Returns:
            Dictionary with score and principal variation
        """
        if not self.engine:
            raise RuntimeError("Engine not started")

        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))

        return {
            'score': info['score'].relative.score(mate_score=10000),
            'pv': info.get('pv', []),
            'depth': info.get('depth', 0)
        }
```

### Phase 6: GUI Implementation

#### Step 6.1: Main Application Window (`src/gui/main_window.py`)
```python
"""
Main application window using Pygame.
Coordinates calibration, game play, and visualization.
"""

import pygame
import sys
from typing import Optional

class MainWindow:
    def __init__(self, config: dict):
        """
        Initialize main window.

        Args:
            config: Configuration dictionary from settings.yaml
        """
        self.config = config

        # Initialize Pygame
        pygame.init()

        # Create window
        if config['gui']['fullscreen']:
            self.screen = pygame.display.set_mode(
                (config['gui']['screen_width'], config['gui']['screen_height']),
                pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode(
                (config['gui']['screen_width'], config['gui']['screen_height'])
            )

        pygame.display.set_caption("P300 Chess BCI")

        # Colors
        self.bg_color = tuple(config['gui']['background_color'])
        self.flash_color = tuple(config['gui']['flash_color'])

        # Clock for timing
        self.clock = pygame.time.Clock()

        # Current screen
        self.current_screen = None

    def run(self):
        """Main application loop."""
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                # Pass events to current screen
                if self.current_screen:
                    self.current_screen.handle_event(event)

            # Update current screen
            if self.current_screen:
                self.current_screen.update()

            # Draw
            self.screen.fill(self.bg_color)
            if self.current_screen:
                self.current_screen.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()
        sys.exit()

    def set_screen(self, screen):
        """Change current screen."""
        self.current_screen = screen
```

#### Step 6.2: Chess Board Visualization (`src/gui/chess_board.py`)
```python
"""
Chess board visualization with move highlighting and flashing.
"""

import pygame
import chess
from typing import List, Set, Optional

class ChessBoardRenderer:
    def __init__(self, board_size: int, position: tuple):
        """
        Initialize chess board renderer.

        Args:
            board_size: Size of board in pixels
            position: (x, y) position of top-left corner
        """
        self.board_size = board_size
        self.position = position
        self.square_size = board_size // 8

        # Colors
        self.light_square = (240, 217, 181)
        self.dark_square = (181, 136, 99)
        self.highlight_color = (255, 255, 100, 128)  # With alpha

        # Current highlights
        self.highlighted_squares = set()
        self.flashing_squares = set()

    def draw(self, surface: pygame.Surface, board: chess.Board):
        """
        Draw the chess board and pieces.

        Args:
            surface: Pygame surface to draw on
            board: Current board state
        """
        # Draw squares
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)  # Flip vertically

                # Determine color
                is_light = (row + col) % 2 == 0
                color = self.light_square if is_light else self.dark_square

                # Draw square
                rect = pygame.Rect(
                    self.position[0] + col * self.square_size,
                    self.position[1] + row * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(surface, color, rect)

                # Flash effect
                if square in self.flashing_squares:
                    flash_surface = pygame.Surface(
                        (self.square_size, self.square_size),
                        pygame.SRCALPHA
                    )
                    flash_surface.fill(self.highlight_color)
                    surface.blit(flash_surface, rect.topleft)

                # Draw piece
                piece = board.piece_at(square)
                if piece:
                    self._draw_piece(surface, piece, rect)

    def _draw_piece(self, surface: pygame.Surface, piece: chess.Piece,
                    rect: pygame.Rect):
        """
        Draw a chess piece (simplified - use images in full implementation).
        """
        # For now, draw text representation
        font = pygame.font.Font(None, self.square_size // 2)

        # Unicode chess pieces
        piece_symbols = {
            chess.PAWN: ('♙', '♟'), chess.KNIGHT: ('♘', '♞'),
            chess.BISHOP: ('♗', '♝'), chess.ROOK: ('♖', '♜'),
            chess.QUEEN: ('♕', '♛'), chess.KING: ('♔', '♚')
        }

        symbol = piece_symbols[piece.piece_type][0 if piece.color else 1]
        text = font.render(symbol, True, (0, 0, 0))
        text_rect = text.get_rect(center=rect.center)
        surface.blit(text, text_rect)

    def set_flashing_squares(self, squares: Set[int]):
        """Set which squares should flash."""
        self.flashing_squares = squares

    def clear_flash(self):
        """Clear all flashing."""
        self.flashing_squares.clear()
```

#### Step 6.3: Flash Controller (`src/gui/flash_controller.py`)
```python
"""
Controls the timing and presentation of visual flashes.
"""

import time
from typing import Callable, List, Optional

class FlashController:
    def __init__(self, flash_duration: float, isi: float):
        """
        Initialize flash controller.

        Args:
            flash_duration: Duration of each flash in seconds
            isi: Inter-stimulus interval in seconds
        """
        self.flash_duration = flash_duration
        self.isi = isi

        # Current state
        self.is_flashing = False
        self.flash_start_time = None
        self.current_flash_index = 0
        self.flash_sequence = []

        # Callbacks
        self.on_flash_start = None  # Called when flash starts
        self.on_flash_end = None    # Called when flash ends

    def start_sequence(self, flash_sequence: List[dict],
                       on_flash_start: Callable = None,
                       on_flash_end: Callable = None):
        """
        Start a flash sequence.

        Args:
            flash_sequence: List of flash events
            on_flash_start: Callback(flash_index, flash_info)
            on_flash_end: Callback(flash_index, flash_info)
        """
        self.flash_sequence = flash_sequence
        self.current_flash_index = 0
        self.on_flash_start = on_flash_start
        self.on_flash_end = on_flash_end

        # Start first flash
        self._start_flash()

    def update(self) -> bool:
        """
        Update flash state.

        Returns:
            True if sequence is still running, False if complete
        """
        if not self.flash_sequence:
            return False

        current_time = time.time()

        if self.is_flashing:
            # Check if flash should end
            if current_time - self.flash_start_time >= self.flash_duration:
                self._end_flash()

        else:
            # Check if ISI has elapsed
            if current_time - self.flash_start_time >= self.flash_duration + self.isi:
                # Move to next flash
                self.current_flash_index += 1

                if self.current_flash_index < len(self.flash_sequence):
                    self._start_flash()
                else:
                    # Sequence complete
                    return False

        return True

    def _start_flash(self):
        """Start a flash."""
        self.is_flashing = True
        self.flash_start_time = time.time()

        if self.on_flash_start:
            flash_info = self.flash_sequence[self.current_flash_index]
            self.on_flash_start(self.current_flash_index, flash_info,
                              self.flash_start_time)

    def _end_flash(self):
        """End current flash."""
        self.is_flashing = False

        if self.on_flash_end:
            flash_info = self.flash_sequence[self.current_flash_index]
            self.on_flash_end(self.current_flash_index, flash_info,
                            time.time())
```

### Phase 7: Main Application Integration

#### Step 7.1: Application Entry Point (`src/main.py`)
```python
"""
Main application entry point.
Coordinates all components and manages application flow.
"""

import yaml
import argparse
from pathlib import Path

from acquisition.lsl_client import LSLClient
from acquisition.data_buffer import DataBuffer
from processing.preprocessing import EEGPreprocessor
from classification.calibration import CalibrationSession
from classification.p300_detector import P300Detector
from chess.game_manager import ChessGameManager
from chess.engine_interface import ChessEngineInterface
from gui.main_window import MainWindow

class P300ChessApp:
    def __init__(self, config_path: str):
        """
        Initialize application.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all application components."""
        # EEG acquisition
        self.lsl_client = LSLClient(
            stream_name=self.config['eeg']['stream_name'],
            expected_channels=self.config['eeg']['channels'],
            sampling_rate=self.config['eeg']['sampling_rate']
        )

        self.data_buffer = DataBuffer(
            n_channels=len(self.config['eeg']['channels']),
            sampling_rate=self.config['eeg']['sampling_rate'],
            buffer_length=self.config['eeg']['buffer_length']
        )

        # Signal processing
        self.preprocessor = EEGPreprocessor(
            sampling_rate=self.config['eeg']['sampling_rate'],
            bandpass=(self.config['processing']['bandpass_low'],
                     self.config['processing']['bandpass_high']),
            notch_freq=self.config['processing']['notch_freq']
        )

        # P300 detection
        self.p300_detector = P300Detector(
            method=self.config['p300']['classifier']
        )

        # Chess components
        self.game_manager = ChessGameManager()
        self.engine = ChessEngineInterface(
            skill_level=self.config['chess']['engine_skill_level'],
            depth=self.config['chess']['engine_depth']
        )

        # GUI
        self.window = MainWindow(self.config)

    def run(self):
        """Run the application."""
        print("P300 Chess BCI Application")
        print("=" * 50)

        # Connect to LSL stream
        print("Connecting to EEG stream...")
        if not self.lsl_client.connect():
            print("Failed to connect to EEG stream")
            return

        # Start data acquisition
        self.lsl_client.start_acquisition(self._on_eeg_sample)

        # Start chess engine
        print("Starting chess engine...")
        self.engine.start()

        # Run calibration
        print("Starting calibration...")
        self._run_calibration()

        # Run main game
        print("Starting game...")
        self._run_game()

        # Cleanup
        self.lsl_client.stop_acquisition()
        self.engine.stop()

    def _on_eeg_sample(self, sample, timestamp):
        """Callback for incoming EEG samples."""
        self.data_buffer.add_sample(sample, timestamp)

    def _run_calibration(self):
        """Run calibration procedure."""
        # TODO: Implement calibration GUI flow
        # This would present calibration stimuli and collect training data
        pass

    def _run_game(self):
        """Run main chess game."""
        # TODO: Implement game flow
        # This would present the chess board, handle move selection,
        # and coordinate with the engine
        pass

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='P300 Chess BCI')
    parser.add_argument('--config', type=str,
                       default='config/settings.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    # Create and run application
    app = P300ChessApp(args.config)
    app.run()

if __name__ == '__main__':
    main()
```

## Implementation Timeline & Phases

### Phase 1: Foundation (Week 1-2)
- Set up project structure
- Implement LSL client and data buffer
- Basic signal processing pipeline
- Unit tests for core components

### Phase 2: P300 Detection (Week 3-4)
- Implement preprocessing and filtering
- Feature extraction methods
- Classifier implementations (LDA, xDAWN)
- Validation with existing P300 datasets

### Phase 3: Calibration System (Week 5-6)
- Calibration GUI with simple stimuli
- Data collection and labeling
- Classifier training pipeline
- Performance metrics and visualization

### Phase 4: Chess Integration (Week 7-8)
- Chess board visualization
- Move generation and validation
- Engine interface (Stockfish)
- Basic game flow

### Phase 5: Move Selection Paradigm (Week 9-10)
- Row/column flashing implementation
- Flash timing and synchronization
- P300-based square selection
- Two-step move selection (origin → destination)

### Phase 6: Integration & Testing (Week 11-12)
- Full system integration
- End-to-end testing
- Performance optimization
- User testing with healthy subjects

### Phase 7: Clinical Validation (Week 13+)
- Testing with target users
- Accessibility improvements
- Documentation
- Deployment

## Technical Considerations

### 1. Timing Precision
- Use high-precision timers for stimulus presentation
- LSL provides timestamp synchronization between EEG and events
- Consider using PsychoPy instead of Pygame for better timing (< 1ms jitter)

### 2. Signal Quality
- Implement real-time signal quality monitoring
- Artifact rejection (eye blinks, muscle activity)
- Adaptive thresholding for noisy environments

### 3. P300 Detection Accuracy
- Typical P300 detection accuracy: 70-90% in good conditions
- May need 5-10 repetitions per selection for reliable detection
- Consider adaptive algorithms that adjust repetitions based on confidence

### 4. User Experience
- Provide clear visual feedback
- Allow adjustable flash speeds and durations
- Implement undo/confirmation for moves
- Show confidence scores to user

### 5. Accessibility
- Large, high-contrast visual elements
- Configurable timing parameters
- Support for different EEG systems (via LSL)
- Error recovery mechanisms

### 6. Performance Optimization
- Real-time processing requirements
- Efficient epoch extraction
- Minimize GUI lag during flashing
- Consider using multiprocessing for heavy computations

## Data Management

### Session Data Storage
```python
# Save session data in HDF5 format
session_data = {
    'metadata': {
        'participant_id': 'P001',
        'date': '2025-01-15',
        'session_type': 'game'
    },
    'calibration': {
        'X': calibration_epochs,
        'y': calibration_labels,
        'classifier_params': params
    },
    'game': {
        'moves': move_history,
        'p300_scores': scores,
        'timestamps': timestamps
    }
}
```

## Testing Strategy

### Unit Tests
- LSL client connection and data acquisition
- Signal processing functions
- Classifier training and prediction
- Chess move validation

### Integration Tests
- End-to-end calibration
- Complete game session
- Engine communication

### User Testing
- Timing accuracy
- Classification performance
- Usability metrics
- Fatigue assessment

## Future Enhancements

### 1. Multiple Engine Support
- Add interface for other engines (Leela, Komodo)
- Engine difficulty selection
- Analysis mode

### 2. Advanced P300 Methods
- Deep learning classifiers (CNN, RNN)
- Transfer learning (reduce calibration time)
- Online adaptation (improve during gameplay)

### 3. Enhanced Move Selection
- Intelligent pre-filtering of legal moves
- Piece-based selection (select piece type first)
- Common opening sequences

### 4. Additional Features
- Game saving/loading
- Online play against humans
- Tutorial mode
- Statistics and progress tracking

### 5. Multi-modal Input
- Hybrid BCI (P300 + motor imagery)
- Eye tracking integration
- EMG for confirmation

## References & Resources

### P300 BCI
- Farwell & Donchin (1988): Original P300 speller
- Rivet et al. (2009): xDAWN spatial filtering
- Congedo et al. (2011): Riemannian geometry for BCI

### Libraries & Tools
- MNE-Python documentation: https://mne.tools/
- Python-chess documentation: https://python-chess.readthedocs.io/
- LSL documentation: https://labstreaminglayer.readthedocs.io/
- Stockfish: https://stockfishchess.org/

### EEG Hardware Compatibility
- OpenBCI (open-source, affordable)
- g.tec (clinical-grade)
- Emotiv (consumer-grade)
- Any LSL-compatible system

## Conclusion

This plan provides a comprehensive roadmap for developing a P300-based BCI chess application. The modular architecture allows for iterative development and easy extension. Key success factors include:

1. Robust signal processing and artifact handling
2. Accurate P300 detection through proper calibration
3. Intuitive visual paradigm for move selection
4. Careful attention to timing and synchronization
5. Extensive testing with target user population

The system can significantly improve quality of life for locked-in syndrome patients by providing an engaging, cognitively stimulating activity that requires only brain signals to play.
