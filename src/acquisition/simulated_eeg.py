"""
Simulated EEG stream for testing P300 BCI without real hardware.
Generates synthetic EEG data with optional P300-like responses.
"""

import pylsl
import numpy as np
import threading
import time
from typing import Optional, List


class PinkNoiseGenerator:
    """
    Generate pink noise (1/f noise) using the Voss-McCartney algorithm.
    Pink noise is more realistic for simulating EEG signals.
    """
    def __init__(self, n_sources: int = 16):
        """
        Initialize pink noise generator.

        Args:
            n_sources: Number of white noise sources to use
        """
        self.n_sources = n_sources
        self.sources = np.zeros(n_sources)
        self.counter = 0

    def next_sample(self) -> float:
        """
        Generate next pink noise sample.

        Returns:
            Pink noise value (approximately N(0,1) distributed)
        """
        # Update sources based on counter bits
        for i in range(self.n_sources):
            if self.counter % (2 ** i) == 0:
                self.sources[i] = np.random.randn()

        self.counter += 1

        # Sum all sources and normalize
        value = np.sum(self.sources) / np.sqrt(self.n_sources)
        return value


class SimulatedEEGStream:
    def __init__(self, stream_name: str = "SimulatedEEG",
                 n_channels: int = 8,
                 sampling_rate: int = 250,
                 channel_names: Optional[List[str]] = None):
        """
        Initialize simulated EEG stream.

        Args:
            stream_name: Name of the LSL stream
            n_channels: Number of EEG channels
            sampling_rate: Sampling rate in Hz
            channel_names: List of channel names
        """
        self.stream_name = stream_name
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names or [f'Ch{i+1}' for i in range(n_channels)]

        self.running = False
        self.thread = None
        self.outlet = None
        self.marker_outlet = None

        # Simulation parameters
        self.base_amplitude = 5.0    # microvolts (base pink noise amplitude - reduced for better SNR)
        self.p300_amplitude = 15.0   # P300 response amplitude (increased for better detection)

        # P300 response characteristics
        self.p300_latency = 0.3  # seconds (300ms)
        self.p300_duration = 0.2  # seconds

        # Current P300 state - support multiple simultaneous P300s
        self.active_p300s = []  # List of (marker, start_time) tuples

        # Pink noise generators (one per channel)
        self.pink_noise_generators = [PinkNoiseGenerator() for _ in range(n_channels)]

        # Target tracking for intelligent P300 generation
        self.target_row = None
        self.target_col = None
        self.p300_reliability = 0.85  # Probability of generating P300 for target (realistic)

    def start(self):
        """Start the simulated EEG stream."""
        if self.running:
            print("Simulated stream already running")
            return

        # Create LSL stream info for EEG data
        info = pylsl.StreamInfo(
            name=self.stream_name,
            type='EEG',
            channel_count=self.n_channels,
            nominal_srate=self.sampling_rate,
            channel_format='float32',
            source_id='simulated_eeg_001'
        )

        # Add channel info
        channels = info.desc().append_child("channels")
        for ch_name in self.channel_names:
            ch = channels.append_child("channel")
            ch.append_child_value("label", ch_name)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")

        # Create outlet
        self.outlet = pylsl.StreamOutlet(info, chunk_size=1, max_buffered=360)

        # Create marker stream for events
        marker_info = pylsl.StreamInfo(
            name=f"{self.stream_name}_Markers",
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='string',
            source_id='simulated_markers_001'
        )
        self.marker_outlet = pylsl.StreamOutlet(marker_info)

        print(f"Starting simulated EEG stream: {self.stream_name}")
        print(f"Channels: {self.n_channels}, Sampling rate: {self.sampling_rate} Hz")

        # Start streaming thread
        self.running = True
        self.thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the simulated stream."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Simulated EEG stream stopped")

    def set_target(self, target_row: Optional[int], target_col: Optional[int]):
        """
        Set the target row and column for P300 generation.

        Args:
            target_row: Target row index (0-7), or None
            target_col: Target column index (0-7), or None
        """
        self.target_row = target_row
        self.target_col = target_col
        if target_row is not None and target_col is not None:
            print(f"SimulatedEEG: Target set to row={target_row}, col={target_col}")

    def clear_target(self):
        """Clear the current target."""
        self.target_row = None
        self.target_col = None

    def send_marker(self, marker: str):
        """
        Send an event marker and intelligently generate P300 for target flashes.

        Args:
            marker: Marker string (e.g., 'flash_row_0', 'flash_col_3')
        """
        if self.marker_outlet:
            self.marker_outlet.push_sample([marker])

            # Intelligently trigger P300 response for target flashes
            should_generate_p300 = False

            if 'flash_row_' in marker and self.target_row is not None:
                # Extract row index from marker
                try:
                    row_idx = int(marker.split('_')[-1])
                    if row_idx == self.target_row:
                        # This is the target row - generate P300 with high probability
                        should_generate_p300 = np.random.random() < self.p300_reliability
                except (ValueError, IndexError):
                    pass

            elif 'flash_col_' in marker and self.target_col is not None:
                # Extract column index from marker
                try:
                    col_idx = int(marker.split('_')[-1])
                    if col_idx == self.target_col:
                        # This is the target column - generate P300 with high probability
                        should_generate_p300 = np.random.random() < self.p300_reliability
                except (ValueError, IndexError):
                    pass

            if should_generate_p300:
                start_time = time.time()
                self.active_p300s.append((marker, start_time))
                print(f"  → P300 generated for {marker}")

    def _streaming_loop(self):
        """Main streaming loop."""
        sample_interval = 1.0 / self.sampling_rate
        next_sample_time = time.time()

        while self.running:
            current_time = time.time()

            if current_time >= next_sample_time:
                # Generate sample
                sample = self._generate_sample(current_time)

                # Push sample
                if self.outlet:
                    self.outlet.push_sample(sample)

                # Schedule next sample
                next_sample_time += sample_interval
            else:
                # Sleep for a short time to avoid busy waiting
                time.sleep(0.001)

    def _generate_sample(self, timestamp: float) -> List[float]:
        """
        Generate a single EEG sample.

        Args:
            timestamp: Current timestamp

        Returns:
            List of channel values in microvolts
        """
        sample = []

        for ch_idx in range(self.n_channels):
            # Generate pink noise (1/f noise - realistic EEG background)
            pink_value = self.pink_noise_generators[ch_idx].next_sample()
            base_signal = pink_value * self.base_amplitude

            # P300 component - sum all active P300s
            p300 = 0.0
            p300s_to_remove = []

            for idx, (marker, start_time) in enumerate(self.active_p300s):
                elapsed = timestamp - start_time

                if self.p300_latency <= elapsed <= (self.p300_latency + self.p300_duration):
                    # Generate P300-like waveform (positive peak around 300ms)
                    relative_time = elapsed - self.p300_latency
                    normalized_time = relative_time / self.p300_duration

                    # Gaussian-like bump
                    p300_component = self.p300_amplitude * np.exp(-((normalized_time - 0.5) ** 2) / 0.1)

                    # Stronger on Pz, Cz channels (assuming they exist)
                    ch_name = self.channel_names[ch_idx]
                    if ch_name in ['Pz', 'Cz', 'P3', 'P4']:
                        p300_component *= 1.5

                    p300 += p300_component

                elif elapsed > (self.p300_latency + self.p300_duration):
                    # P300 finished - mark for removal
                    p300s_to_remove.append(idx)

            # Remove completed P300s (do this once per sample, not per channel)
            if ch_idx == 0:  # Only remove once per sample
                for idx in reversed(p300s_to_remove):
                    self.active_p300s.pop(idx)

            # Combine components
            value = base_signal + p300
            sample.append(float(value))

        return sample

    def is_running(self) -> bool:
        """Check if stream is running."""
        return self.running


def main():
    """Test the simulated EEG stream."""
    print("Starting simulated EEG stream for testing...")

    # Create simulated stream with standard 10-20 channels
    sim = SimulatedEEGStream(
        stream_name="SimulatedEEG",
        n_channels=8,
        sampling_rate=250,
        channel_names=['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'C3', 'C4']
    )

    sim.start()

    print("\nSimulated stream is running...")
    print("Press Ctrl+C to stop\n")

    # Send some test markers
    try:
        marker_count = 0
        while True:
            time.sleep(2.0)
            marker = f"test_marker_{marker_count}"
            sim.send_marker(marker)
            print(f"Sent marker: {marker}")
            marker_count += 1
    except KeyboardInterrupt:
        print("\nStopping...")
        sim.stop()


if __name__ == '__main__':
    main()
