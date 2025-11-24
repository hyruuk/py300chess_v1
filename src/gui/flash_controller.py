"""
Controls the timing and presentation of visual flashes.
"""

import time
from typing import Callable, List, Optional, Dict, Tuple


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
        self.on_sequence_complete = None  # Called when sequence finishes

        # Timing
        self.last_event_time = None
        self.is_running = False

    def start_sequence(self, flash_sequence: List[Dict],
                       on_flash_start: Callable = None,
                       on_flash_end: Callable = None,
                       on_sequence_complete: Callable = None):
        """
        Start a flash sequence.

        Args:
            flash_sequence: List of flash events (dictionaries with flash info)
            on_flash_start: Callback(flash_index, flash_info, timestamp)
            on_flash_end: Callback(flash_index, flash_info, timestamp)
            on_sequence_complete: Callback() when sequence finishes
        """
        self.flash_sequence = flash_sequence
        self.current_flash_index = 0
        self.on_flash_start = on_flash_start
        self.on_flash_end = on_flash_end
        self.on_sequence_complete = on_sequence_complete

        self.is_running = True

        # Start first flash immediately
        self._start_flash()

    def update(self) -> bool:
        """
        Update flash state. Should be called every frame.

        Returns:
            True if sequence is still running, False if complete
        """
        if not self.is_running:
            return False

        current_time = time.time()

        if self.is_flashing:
            # Check if flash should end
            if current_time - self.flash_start_time >= self.flash_duration:
                self._end_flash()

        else:
            # Check if ISI has elapsed and we should start next flash
            if self.last_event_time and \
               current_time - self.last_event_time >= self.isi:
                # Move to next flash
                self.current_flash_index += 1

                if self.current_flash_index < len(self.flash_sequence):
                    self._start_flash()
                else:
                    # Sequence complete
                    self._complete_sequence()
                    return False

        return True

    def _start_flash(self):
        """Start a flash."""
        self.is_flashing = True
        self.flash_start_time = time.time()
        self.last_event_time = self.flash_start_time

        if self.on_flash_start:
            flash_info = self.flash_sequence[self.current_flash_index]
            self.on_flash_start(self.current_flash_index, flash_info,
                              self.flash_start_time)

    def _end_flash(self):
        """End current flash."""
        self.is_flashing = False
        end_time = time.time()
        self.last_event_time = end_time

        if self.on_flash_end:
            flash_info = self.flash_sequence[self.current_flash_index]
            self.on_flash_end(self.current_flash_index, flash_info, end_time)

    def _complete_sequence(self):
        """Complete the flash sequence."""
        self.is_running = False

        if self.on_sequence_complete:
            self.on_sequence_complete()

    def stop(self):
        """Stop the current sequence."""
        self.is_running = False
        self.is_flashing = False

    def is_active(self) -> bool:
        """Check if flash sequence is active."""
        return self.is_running

    def get_progress(self) -> Tuple[int, int]:
        """
        Get progress through sequence.

        Returns:
            (current_index, total_flashes)
        """
        return self.current_flash_index, len(self.flash_sequence)

    def get_current_flash(self) -> Optional[Dict]:
        """
        Get current flash information.

        Returns:
            Current flash info or None
        """
        if self.is_running and self.current_flash_index < len(self.flash_sequence):
            return self.flash_sequence[self.current_flash_index]
        return None

    def get_timing_info(self) -> Dict:
        """
        Get timing information.

        Returns:
            Dictionary with timing info
        """
        total_time = len(self.flash_sequence) * (self.flash_duration + self.isi)

        return {
            'flash_duration': self.flash_duration,
            'isi': self.isi,
            'total_flashes': len(self.flash_sequence),
            'current_flash': self.current_flash_index,
            'estimated_time_remaining': (len(self.flash_sequence) - self.current_flash_index) *
                                       (self.flash_duration + self.isi),
            'total_time': total_time
        }

    def reset(self):
        """Reset controller to initial state."""
        self.is_flashing = False
        self.flash_start_time = None
        self.current_flash_index = 0
        self.flash_sequence = []
        self.last_event_time = None
        self.is_running = False
