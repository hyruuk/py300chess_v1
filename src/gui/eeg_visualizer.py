"""
Real-time EEG visualization component with scrolling time series display.
"""

import pygame
import numpy as np
from collections import deque
from typing import List, Optional, Tuple
import time


class EEGVisualizer:
    def __init__(self, n_channels: int, channel_names: List[str],
                 sampling_rate: int, position: Tuple[int, int],
                 width: int, height: int, window_duration: float = 5.0):
        """
        Initialize EEG visualizer component.

        Args:
            n_channels: Number of EEG channels
            channel_names: List of channel names
            sampling_rate: Sampling rate in Hz
            position: (x, y) position to draw the component
            width: Component width in pixels
            height: Component height in pixels
            window_duration: Duration of data to display (seconds)
        """
        self.n_channels = n_channels
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate
        self.position = position
        self.width = width
        self.height = height
        self.window_duration = window_duration

        # Calculate buffer size
        self.buffer_size = int(window_duration * sampling_rate)

        # Data buffers for each channel
        self.data_buffers = [deque(maxlen=self.buffer_size) for _ in range(n_channels)]

        # Display parameters
        self.channel_height = (height - 60) // n_channels
        self.amplitude_scale = 1.5  # Pixels per microvolt

        # Colors
        self.bg_color = (20, 20, 25)
        self.grid_color = (40, 40, 50)
        self.text_color = (180, 180, 180)
        self.channel_colors = [
            (100, 200, 255),  # Light blue
            (255, 200, 100),  # Orange
            (100, 255, 150),  # Green
            (255, 150, 200),  # Pink
            (200, 150, 255),  # Purple
            (255, 255, 100),  # Yellow
            (150, 255, 255),  # Cyan
            (255, 150, 100),  # Coral
        ]

        # Fonts (will be initialized when draw is called)
        self.font = None
        self.small_font = None

        # Markers
        self.markers = []  # List of (timestamp, marker_text, color)
        self.marker_display_duration = 2.0  # seconds

    def add_sample(self, sample: np.ndarray, timestamp: float):
        """
        Add a new EEG sample to the visualization.

        Args:
            sample: EEG sample (n_channels,)
            timestamp: Sample timestamp
        """
        for ch_idx, value in enumerate(sample):
            if ch_idx < self.n_channels:
                self.data_buffers[ch_idx].append((timestamp, value))

    def add_marker(self, marker_text: str, color: tuple = (255, 100, 100)):
        """
        Add an event marker to display.

        Args:
            marker_text: Marker text
            color: Marker color (RGB tuple)
        """
        self.markers.append((time.time(), marker_text, color))

    def draw(self, surface: pygame.Surface):
        """
        Draw the EEG visualization on the given surface.

        Args:
            surface: Pygame surface to draw on
        """
        # Initialize fonts if needed
        if self.font is None:
            self.font = pygame.font.Font(None, 20)
            self.small_font = pygame.font.Font(None, 16)

        # Draw background
        bg_rect = pygame.Rect(self.position[0], self.position[1], self.width, self.height)
        pygame.draw.rect(surface, self.bg_color, bg_rect)
        pygame.draw.rect(surface, (60, 60, 70), bg_rect, 2)

        # Draw title
        title = self.font.render("EEG Signals", True, self.text_color)
        surface.blit(title, (self.position[0] + 10, self.position[1] + 5))

        # Draw each channel
        for ch_idx in range(self.n_channels):
            self._draw_channel(surface, ch_idx)

        # Draw markers
        self._draw_markers(surface)

        # Draw scale info
        scale_text = f"{self.amplitude_scale:.1f} px/µV"
        scale_surface = self.small_font.render(scale_text, True, (100, 100, 100))
        surface.blit(scale_surface, (self.position[0] + self.width - 80,
                                     self.position[1] + self.height - 20))

    def _draw_channel(self, surface: pygame.Surface, ch_idx: int):
        """Draw a single channel's time series."""
        if not self.data_buffers[ch_idx]:
            return

        # Calculate channel position (relative to component position)
        y_baseline = self.position[1] + 35 + ch_idx * self.channel_height + self.channel_height // 2

        # Drawing area
        left_margin = self.position[0] + 40
        right_margin = self.position[0] + self.width - 10
        plot_width = right_margin - left_margin

        # Draw baseline
        pygame.draw.line(surface, self.grid_color,
                        (left_margin, y_baseline), (right_margin, y_baseline), 1)

        # Draw channel label
        color = self.channel_colors[ch_idx % len(self.channel_colors)]
        label = self.small_font.render(self.channel_names[ch_idx], True, color)
        surface.blit(label, (self.position[0] + 5, y_baseline - 8))

        # Get data
        data = list(self.data_buffers[ch_idx])
        if len(data) < 2:
            return

        # Get time range
        current_time = data[-1][0] if data else 0
        start_time = current_time - self.window_duration

        # Convert to screen coordinates
        points = []
        for timestamp, value in data:
            # X position (time)
            time_offset = timestamp - start_time
            x = left_margin + int((time_offset / self.window_duration) * plot_width)

            # Y position (amplitude)
            y = y_baseline - int(value * self.amplitude_scale)

            # Clamp Y to channel bounds
            ch_top = self.position[1] + 35 + ch_idx * self.channel_height
            ch_bottom = ch_top + self.channel_height
            y = max(ch_top, min(ch_bottom - 1, y))

            points.append((x, y))

        # Draw the signal
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 1)

    def _draw_markers(self, surface: pygame.Surface):
        """Draw event markers."""
        current_time = time.time()

        # Filter out old markers
        self.markers = [(t, text, color) for t, text, color in self.markers
                       if current_time - t < self.marker_display_duration]

        # Draw active markers (stacked vertically)
        marker_x = self.position[0] + self.width - 100
        marker_y = self.position[1] + 30

        for marker_time, marker_text, marker_color in self.markers[-3:]:  # Show last 3
            # Calculate fade based on age
            age = current_time - marker_time
            alpha = int(255 * (1.0 - age / self.marker_display_duration))
            alpha = max(0, min(255, alpha))

            # Create text with fade
            text_surface = self.small_font.render(marker_text, True, marker_color)
            text_surface.set_alpha(alpha)

            # Draw text
            surface.blit(text_surface, (marker_x, marker_y))
            marker_y += 20
