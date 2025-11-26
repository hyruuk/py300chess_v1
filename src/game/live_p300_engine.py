"""
Live P300 processing engine for real-time move selection.
Processes incoming EEG data and flash markers to detect P300 responses.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict


class LiveP300Engine:
    def __init__(self, data_buffer, preprocessor, feature_extractor,
                 p300_detector=None, use_random_selection=False):
        """
        Initialize live P300 engine.

        Args:
            data_buffer: DataBuffer instance for accessing EEG data
            preprocessor: EEGPreprocessor for signal processing
            feature_extractor: P300FeatureExtractor for feature extraction
            p300_detector: Trained P300Detector (optional, can be None for testing)
            use_random_selection: If True, use random selection instead of P300 detection
        """
        self.data_buffer = data_buffer
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.p300_detector = p300_detector
        self.use_random_selection = use_random_selection

        # Flash tracking
        self.flash_events = []  # List of flash events with timestamps
        self.current_sequence_id = None
        self.sequence_active = False

        # Epoch extraction parameters (from config)
        self.epoch_tmin = -0.2  # seconds before flash
        self.epoch_tmax = 0.8   # seconds after flash

        # P300 scores for current sequence
        self.flash_scores = {}  # {flash_index: score}

        # Callbacks
        self.on_sequence_complete = None

        # Thread safety
        self.lock = threading.Lock()

    def start_sequence(self, sequence_id: str, n_flashes: int,
                      on_complete: Optional[Callable] = None):
        """
        Start a new flash sequence.

        Args:
            sequence_id: Unique identifier for this sequence
            n_flashes: Expected number of flashes in sequence
            on_complete: Callback when sequence is complete
        """
        with self.lock:
            self.current_sequence_id = sequence_id
            self.sequence_active = True
            self.flash_events = []
            self.flash_scores = {}
            self.on_sequence_complete = on_complete

            mode = "RANDOM" if self.use_random_selection else "REAL DETECTION"
            trained = "TRAINED" if (self.p300_detector and self.p300_detector.is_trained) else "UNTRAINED"
            print(f"LiveP300Engine: Started sequence {sequence_id} | Mode: {mode} | Detector: {trained} | {n_flashes} flashes")

    def add_flash_event(self, flash_index: int, flash_info: Dict, timestamp: float):
        """
        Record a flash event.

        Args:
            flash_index: Index of this flash in the sequence
            flash_info: Dictionary with flash information (type, index, squares, etc.)
            timestamp: LSL timestamp when flash occurred
        """
        with self.lock:
            if not self.sequence_active:
                return

            event = {
                'flash_index': flash_index,
                'flash_info': flash_info,
                'timestamp': timestamp,
                'processed': False
            }

            self.flash_events.append(event)

            # Add marker to data buffer
            self.data_buffer.add_event(
                timestamp=timestamp,
                event_code=flash_index,
                event_info=flash_info
            )

    def process_pending_flashes(self):
        """
        Process any flashes that haven't been analyzed yet.
        Should be called periodically to process accumulated data.
        """
        with self.lock:
            if not self.sequence_active:
                return

            current_time = time.time()

            for event in self.flash_events:
                if event['processed']:
                    continue

                flash_timestamp = event['timestamp']
                time_since_flash = current_time - flash_timestamp

                # Wait until we have enough data after the flash
                # Need data from tmin to tmax, so wait for tmax + safety margin
                required_wait = self.epoch_tmax + 0.5  # Increased safety margin
                if time_since_flash >= required_wait:
                    # Process this flash
                    score = self._process_flash(event)
                    self.flash_scores[event['flash_index']] = score
                    event['processed'] = True

    def end_sequence(self) -> Dict[str, np.ndarray]:
        """
        End the current sequence and return results.

        Returns:
            Dictionary with results including scores by row/column
        """
        with self.lock:
            if not self.sequence_active:
                return None

            # Process any remaining flashes
            self._wait_for_all_flashes()

            # Aggregate scores by row and column
            results = self._aggregate_scores()

            # Reset state
            self.sequence_active = False
            self.current_sequence_id = None

            # Call completion callback
            if self.on_sequence_complete:
                self.on_sequence_complete(results)

            return results

    def _process_flash(self, event: Dict) -> float:
        """
        Process a single flash event and return P300 score.

        Args:
            event: Flash event dictionary

        Returns:
            P300 score (higher = more likely P300)
        """
        flash_timestamp = event['timestamp']

        if self.use_random_selection:
            # Random selection for testing (used during calibration before training)
            score = np.random.random()
            return float(score)

        # Validate buffer has enough data before extracting
        if self.data_buffer.samples_written < 10:  # At least 10 samples
            print(f"Warning: Buffer has insufficient samples ({self.data_buffer.samples_written})")
            return 0.0

        # Extract epoch around flash
        epoch = self.data_buffer.get_epoch(
            event_timestamp=flash_timestamp,
            tmin=self.epoch_tmin,
            tmax=self.epoch_tmax
        )

        if epoch is None or len(epoch) == 0:
            print(f"Warning: Could not extract epoch for flash {event['flash_index']}")
            print(f"  Flash timestamp: {flash_timestamp:.3f}")
            print(f"  Buffer samples: {self.data_buffer.samples_written}")
            return 0.0

        # Debug: check epoch shape for first few flashes
        if event['flash_index'] < 3:
            print(f"DEBUG Flash {event['flash_index']}: epoch shape: {epoch.shape}, expected: (250, 8)")
            print(f"  Flash timestamp: {flash_timestamp:.3f}")
            print(f"  Epoch window: [{flash_timestamp + self.epoch_tmin:.3f}, {flash_timestamp + self.epoch_tmax:.3f}]")
            print(f"  Time since flash: {time.time() - flash_timestamp:.3f}s")

        # Preprocess epoch
        try:
            epoch_preprocessed = self.preprocessor.filter(epoch)

            # Get P300 prediction (detector extracts features internally)
            if self.p300_detector and self.p300_detector.is_trained:
                prediction, confidence = self.p300_detector.predict(epoch_preprocessed)
                score = confidence
            else:
                # No trained detector - use random
                score = np.random.random()

            return float(score)

        except Exception as e:
            print(f"Error processing flash {event['flash_index']}: {e}")
            return 0.0

    def _wait_for_all_flashes(self):
        """Wait for all flashes to be processed."""
        max_wait = 2.0  # Maximum wait time in seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            all_processed = all(event['processed'] for event in self.flash_events)
            if all_processed:
                break

            # Process any pending flashes
            self.process_pending_flashes()
            time.sleep(0.05)

    def _aggregate_scores(self) -> Dict:
        """
        Aggregate flash scores by row and column.

        Returns:
            Dictionary with aggregated scores
        """
        row_scores = defaultdict(list)
        col_scores = defaultdict(list)

        for flash_index, score in self.flash_scores.items():
            # Find corresponding flash event
            event = next((e for e in self.flash_events
                         if e['flash_index'] == flash_index), None)

            if event:
                flash_info = event['flash_info']
                flash_type = flash_info.get('type')
                flash_idx = flash_info.get('index')

                if flash_type == 'row':
                    row_scores[flash_idx].append(score)
                elif flash_type == 'col':
                    col_scores[flash_idx].append(score)

        # Average scores
        row_scores_avg = np.zeros(8)
        col_scores_avg = np.zeros(8)

        for row_idx, scores in row_scores.items():
            row_scores_avg[row_idx] = np.mean(scores)

        for col_idx, scores in col_scores.items():
            col_scores_avg[col_idx] = np.mean(scores)

        return {
            'row_scores': row_scores_avg,
            'col_scores': col_scores_avg,
            'raw_scores': self.flash_scores,
            'n_flashes_processed': len(self.flash_scores)
        }

    def is_active(self) -> bool:
        """Check if a sequence is currently active."""
        with self.lock:
            return self.sequence_active

    def get_progress(self) -> Tuple[int, int]:
        """
        Get progress of current sequence.

        Returns:
            (processed_count, total_count)
        """
        with self.lock:
            if not self.sequence_active:
                return (0, 0)

            processed = sum(1 for e in self.flash_events if e['processed'])
            total = len(self.flash_events)

            return (processed, total)

    def is_processing_complete(self) -> bool:
        """
        Check if all flashes have been processed without blocking.

        Returns:
            True if all flashes are processed, False otherwise
        """
        with self.lock:
            if not self.sequence_active:
                return True

            if not self.flash_events:
                return True

            return all(event['processed'] for event in self.flash_events)

    def end_sequence_nonblocking(self) -> Optional[Dict[str, np.ndarray]]:
        """
        End sequence without waiting for remaining flashes.
        Only returns results if all flashes are already processed.

        Returns:
            Dictionary with results, or None if not ready
        """
        with self.lock:
            if not self.sequence_active:
                return None

            # Check if all flashes are processed
            if not all(event['processed'] for event in self.flash_events):
                return None

            # Aggregate scores by row and column
            results = self._aggregate_scores()

            # Reset state
            self.sequence_active = False
            self.current_sequence_id = None

            return results

    def reset(self):
        """Reset the engine state."""
        with self.lock:
            self.sequence_active = False
            self.current_sequence_id = None
            self.flash_events = []
            self.flash_scores = {}
