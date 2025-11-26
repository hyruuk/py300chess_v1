"""
Circular buffer for storing incoming EEG data with event markers.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict


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
        """
        Add a new sample to the buffer.

        Args:
            sample: EEG sample (n_channels,)
            timestamp: LSL timestamp for this sample
        """
        self.data[self.write_index] = sample
        self.timestamps[self.write_index] = timestamp
        self.write_index = (self.write_index + 1) % self.buffer_size
        self.samples_written += 1

    def add_event(self, timestamp: float, event_code: int,
                  event_info: dict = None):
        """
        Add an event marker.

        Args:
            timestamp: Event timestamp
            event_code: Numerical event code
            event_info: Additional event information
        """
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

        # Get valid data from circular buffer
        if self.samples_written < self.buffer_size:
            # Buffer not yet full - data is contiguous from 0 to write_index
            valid_timestamps = self.timestamps[:self.write_index]
            valid_data = self.data[:self.write_index]
        else:
            # Buffer has wrapped - need to unwrap it to get chronological order
            # Most recent data is at [write_index : buffer_size] + [0 : write_index]
            # Oldest to newest: [write_index : buffer_size] then [0 : write_index]
            valid_timestamps = np.concatenate([
                self.timestamps[self.write_index:],
                self.timestamps[:self.write_index]
            ])
            valid_data = np.vstack([
                self.data[self.write_index:],
                self.data[:self.write_index]
            ])

        # Find samples in time window
        mask = (valid_timestamps >= start_time) & (valid_timestamps <= end_time)
        epoch_data = valid_data[mask]

        if len(epoch_data) == 0:
            return None

        # Debug: check if epoch is unexpectedly short
        expected_samples = int((tmax - tmin) * 250)  # Assuming 250Hz
        if len(epoch_data) < expected_samples * 0.8:  # Less than 80% of expected
            print(f"WARNING: Short epoch extracted - got {len(epoch_data)} samples, expected ~{expected_samples}")
            print(f"  Time window: [{start_time:.3f}, {end_time:.3f}] (duration: {end_time-start_time:.3f}s)")
            print(f"  Buffer range: [{valid_timestamps.min():.3f}, {valid_timestamps.max():.3f}]")
            print(f"  Samples in window: {np.sum(mask)}")

        return epoch_data

    def get_latest_data(self, duration: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the most recent data from the buffer.

        Args:
            duration: Duration of data to retrieve (seconds)

        Returns:
            Tuple of (data, timestamps) or None if insufficient data
        """
        n_samples = int(duration * self.sampling_rate)

        if self.samples_written < n_samples:
            return None

        if self.samples_written < self.buffer_size:
            # Buffer not yet full
            data = self.data[max(0, self.write_index - n_samples):self.write_index]
            timestamps = self.timestamps[max(0, self.write_index - n_samples):self.write_index]
        else:
            # Buffer is full, handle wraparound
            if n_samples <= self.write_index:
                data = self.data[self.write_index - n_samples:self.write_index]
                timestamps = self.timestamps[self.write_index - n_samples:self.write_index]
            else:
                # Need to wrap around
                first_part = self.buffer_size - (n_samples - self.write_index)
                data = np.vstack([
                    self.data[first_part:],
                    self.data[:self.write_index]
                ])
                timestamps = np.concatenate([
                    self.timestamps[first_part:],
                    self.timestamps[:self.write_index]
                ])

        return data, timestamps

    def clear_events(self):
        """Clear all event markers."""
        self.events = []

    def get_events(self) -> List[Dict]:
        """Get all event markers."""
        return self.events.copy()
