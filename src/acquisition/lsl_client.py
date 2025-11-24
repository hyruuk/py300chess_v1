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

        Args:
            timeout: Maximum time to wait for stream (seconds)

        Returns:
            True if connection successful, False otherwise
        """
        print(f"Resolving LSL stream: {self.stream_name}...")

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
                                       args=(callback,), daemon=True)
        self.thread.start()
        print("Data acquisition started")

    def _acquisition_loop(self, callback: Callable):
        """Background acquisition loop."""
        while self.running:
            sample, timestamp = self.inlet.pull_sample(timeout=0.0)
            if sample:
                callback(np.array(sample), timestamp)
            else:
                time.sleep(0.001)  # Small sleep to avoid busy waiting

    def stop_acquisition(self):
        """Stop data acquisition."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Data acquisition stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_acquisition()
