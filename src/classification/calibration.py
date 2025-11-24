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
        self.event_timestamps = []

    def _generate_trials(self) -> List[dict]:
        """Generate randomized trial sequence."""
        trials = []

        for target_id in range(self.n_targets):
            for trial in range(self.n_trials_per_target):
                # Create a sequence with target and non-targets
                sequence = [{'stimulus_id': target_id, 'is_target': True}]

                # Add non-targets (other stimuli)
                nontarget_ids = [i for i in range(self.n_targets) if i != target_id]

                if len(nontarget_ids) >= self.n_nontargets_per_target:
                    selected_nontargets = random.sample(nontarget_ids,
                                                       self.n_nontargets_per_target)
                else:
                    # If not enough unique non-targets, repeat some
                    selected_nontargets = random.choices(nontarget_ids,
                                                        k=self.n_nontargets_per_target)

                for nt_id in selected_nontargets:
                    sequence.append({'stimulus_id': nt_id, 'is_target': False})

                # Randomize sequence
                random.shuffle(sequence)

                trials.append({
                    'target_id': target_id,
                    'sequence': sequence,
                    'trial_number': trial
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

    def is_complete(self) -> bool:
        """Check if calibration is complete."""
        return self.current_trial >= len(self.trials)

    def get_progress(self) -> Tuple[int, int]:
        """Get calibration progress."""
        return self.current_trial, len(self.trials)

    def add_epoch(self, epoch: np.ndarray, is_target: bool, timestamp: float = None):
        """
        Add a labeled epoch to training data.

        Args:
            epoch: EEG epoch data
            is_target: Whether this was a target stimulus
            timestamp: Optional timestamp for this epoch
        """
        self.epochs.append(epoch)
        self.labels.append(1 if is_target else 0)
        if timestamp is not None:
            self.event_timestamps.append(timestamp)

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get collected training data.

        Returns:
            Tuple of (X, y) where X is epochs array and y is labels array
        """
        X = np.array(self.epochs)
        y = np.array(self.labels)
        return X, y

    def get_statistics(self) -> dict:
        """
        Get calibration statistics.

        Returns:
            Dictionary with statistics
        """
        y = np.array(self.labels)
        return {
            'total_epochs': len(self.epochs),
            'n_targets': np.sum(y == 1),
            'n_nontargets': np.sum(y == 0),
            'trials_completed': self.current_trial,
            'trials_total': len(self.trials)
        }

    def reset(self):
        """Reset calibration session."""
        self.trials = self._generate_trials()
        self.current_trial = 0
        self.epochs = []
        self.labels = []
        self.event_timestamps = []
