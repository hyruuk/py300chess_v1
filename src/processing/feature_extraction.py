"""
Feature extraction for P300 classification.
"""

import numpy as np
from typing import List, Optional, Tuple


class P300FeatureExtractor:
    def __init__(self, target_channels: List[int], sampling_rate: int,
                 downsample_factor: int = 4):
        """
        Initialize feature extractor.

        Args:
            target_channels: Indices of channels to use for features
            sampling_rate: Sampling rate in Hz
            downsample_factor: Factor to downsample by (e.g., 4 for 250Hz -> ~60Hz)
        """
        self.target_channels = target_channels
        self.sampling_rate = sampling_rate
        self.downsample_factor = downsample_factor

    def extract_temporal_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract temporal features (downsampled epoch).

        Args:
            epoch: Single epoch (n_samples, n_channels)

        Returns:
            Feature vector
        """
        # Select target channels
        if epoch.ndim == 1:
            # Single channel case
            epoch_subset = epoch
        else:
            epoch_subset = epoch[:, self.target_channels]

        # Downsample
        downsampled = epoch_subset[::self.downsample_factor]

        # Flatten to feature vector
        features = downsampled.flatten() if downsampled.ndim > 1 else downsampled

        return features

    def extract_spatial_features(self, epoch: np.ndarray,
                                  spatial_filters: np.ndarray) -> np.ndarray:
        """
        Extract spatial features using learned filters (e.g., xDAWN).

        Args:
            epoch: Single epoch (n_samples, n_channels)
            spatial_filters: Spatial filter matrix (n_channels, n_components)

        Returns:
            Spatially filtered feature vector
        """
        # Apply spatial filters
        filtered = epoch @ spatial_filters

        # Downsample and flatten
        downsampled = filtered[::self.downsample_factor, :]
        features = downsampled.flatten()

        return features

    def extract_spectral_features(self, epoch: np.ndarray,
                                   freq_bands: List[Tuple[float, float]]) -> np.ndarray:
        """
        Extract spectral features (power in frequency bands).

        Args:
            epoch: Single epoch (n_samples, n_channels)
            freq_bands: List of (low, high) frequency bands in Hz

        Returns:
            Feature vector with power in each band
        """
        from scipy import signal as sp_signal

        features = []

        # For each channel
        for ch_idx in self.target_channels:
            if epoch.ndim == 1:
                ch_data = epoch
            else:
                ch_data = epoch[:, ch_idx]

            # Compute power spectral density
            freqs, psd = sp_signal.welch(ch_data, fs=self.sampling_rate,
                                         nperseg=min(256, len(ch_data)))

            # Extract power in each frequency band
            for low, high in freq_bands:
                freq_mask = (freqs >= low) & (freqs <= high)
                band_power = np.mean(psd[freq_mask])
                features.append(band_power)

        return np.array(features)

    def extract_combined_features(self, epoch: np.ndarray,
                                   spatial_filters: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract combined temporal and spatial features.

        Args:
            epoch: Single epoch (n_samples, n_channels)
            spatial_filters: Optional spatial filters

        Returns:
            Combined feature vector
        """
        temporal = self.extract_temporal_features(epoch)

        if spatial_filters is not None:
            spatial = self.extract_spatial_features(epoch, spatial_filters)
            return np.concatenate([temporal, spatial])
        else:
            return temporal
