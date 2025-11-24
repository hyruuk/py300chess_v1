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
            notch_freq: Notch filter frequency (50 or 60 Hz), None to disable
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

        # Ensure frequencies are in valid range (0, 1)
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))

        if low >= high:
            raise ValueError(f"Invalid bandpass range: {self.bandpass}")

        self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')

        # Notch filter (if specified)
        if self.notch_freq:
            w0 = self.notch_freq / nyquist
            Q = 30  # Quality factor
            self.notch_b, self.notch_a = signal.iirnotch(w0, Q)
        else:
            self.notch_b, self.notch_a = None, None

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filtering to data.

        Args:
            data: Input data (n_samples, n_channels) or (n_samples,)

        Returns:
            Filtered data
        """
        if data.ndim == 1:
            # Single channel
            filtered = signal.filtfilt(self.bp_b, self.bp_a, data)
            if self.notch_freq:
                filtered = signal.filtfilt(self.notch_b, self.notch_a, filtered)
            return filtered

        # Multiple channels
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

    def baseline_correct(self, epoch: np.ndarray,
                        baseline_window: Tuple[int, int]) -> np.ndarray:
        """
        Apply baseline correction.

        Args:
            epoch: Epoch data (n_samples, n_channels) or (n_samples,)
            baseline_window: (start, end) sample indices for baseline

        Returns:
            Baseline-corrected epoch
        """
        if epoch.ndim == 1:
            # Single channel
            baseline = epoch[baseline_window[0]:baseline_window[1]].mean()
            return epoch - baseline

        # Multiple channels
        baseline = epoch[baseline_window[0]:baseline_window[1], :].mean(axis=0)
        return epoch - baseline

    def epoch_data(self, data: np.ndarray, timestamps: np.ndarray,
                   event_time: float, tmin: float, tmax: float) -> Optional[np.ndarray]:
        """
        Extract an epoch from continuous data.

        Args:
            data: Continuous data (n_samples, n_channels)
            timestamps: Timestamps for each sample (n_samples,)
            event_time: Event timestamp
            tmin: Start time relative to event (seconds, negative)
            tmax: End time relative to event (seconds, positive)

        Returns:
            Epoch data (n_epoch_samples, n_channels) or None if not enough data
        """
        # Find samples in time window
        start_time = event_time + tmin
        end_time = event_time + tmax

        mask = (timestamps >= start_time) & (timestamps <= end_time)

        if np.sum(mask) == 0:
            return None

        epoch = data[mask]

        # Check if we have enough samples
        expected_samples = int((tmax - tmin) * self.sampling_rate)
        if len(epoch) < expected_samples * 0.9:  # Allow 10% tolerance
            return None

        return epoch

    def get_baseline_window(self, tmin: float, baseline_end: float = 0.0) -> Tuple[int, int]:
        """
        Get baseline window in sample indices.

        Args:
            tmin: Epoch start time (seconds, negative)
            baseline_end: Baseline end time (seconds, typically 0)

        Returns:
            (start_idx, end_idx) in samples
        """
        start_idx = 0
        end_idx = int((baseline_end - tmin) * self.sampling_rate)
        return start_idx, end_idx
