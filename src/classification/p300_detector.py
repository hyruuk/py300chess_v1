"""
P300 detection classifier with multiple algorithm options.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from typing import Tuple, Optional
import pickle


class P300Detector:
    def __init__(self, method: str = 'LDA', n_xdawn_components: int = 4):
        """
        Initialize P300 detector.

        Args:
            method: Classification method ('LDA', 'xDAWN_LDA')
            n_xdawn_components: Number of xDAWN spatial filters
        """
        self.method = method
        self.n_xdawn_components = n_xdawn_components
        self.classifier = None
        self.spatial_filters = None
        self.is_trained = False
        self.training_metrics = {}

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the classifier.

        Args:
            X: Training epochs (n_epochs, n_samples, n_channels)
            y: Labels (0=non-target, 1=target)

        Returns:
            Dictionary with training metrics
        """
        print(f"Training P300 detector using {self.method}...")
        print(f"Data shape: {X.shape}, Labels: {y.shape}")
        print(f"Targets: {np.sum(y == 1)}, Non-targets: {np.sum(y == 0)}")

        # Flatten epochs to feature vectors
        n_epochs = X.shape[0]
        X_flat = X.reshape(n_epochs, -1)

        if self.method == 'LDA':
            self.classifier = LinearDiscriminantAnalysis(solver='lsqr',
                                                         shrinkage='auto')
            X_train = X_flat

        elif self.method == 'xDAWN_LDA':
            # Compute xDAWN spatial filters
            print("Computing xDAWN spatial filters...")
            self.spatial_filters = self._compute_xdawn_filters(X, y,
                                                               self.n_xdawn_components)
            # Apply spatial filtering
            X_filtered = self._apply_spatial_filters(X)
            X_train = X_filtered.reshape(n_epochs, -1)

            self.classifier = LinearDiscriminantAnalysis(solver='lsqr',
                                                         shrinkage='auto')

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Train classifier
        self.classifier.fit(X_train, y)

        # Cross-validation score
        print("Performing cross-validation...")
        cv_scores = cross_val_score(self.classifier, X_train, y, cv=5)

        self.is_trained = True

        self.training_metrics = {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_targets': int(np.sum(y == 1)),
            'n_nontargets': int(np.sum(y == 0)),
            'method': self.method
        }

        print(f"Training complete! CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

        return self.training_metrics

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
        proba = self.classifier.predict_proba([features])[0]
        confidence = float(proba[1])  # Probability of class 1 (target)

        return int(prediction), confidence

    def predict_batch(self, epochs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for multiple epochs.

        Args:
            epochs: Multiple epochs (n_epochs, n_samples, n_channels)

        Returns:
            (predictions, confidences)
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained")

        n_epochs = epochs.shape[0]

        # Apply spatial filters if using xDAWN
        if self.spatial_filters is not None:
            epochs_filtered = self._apply_spatial_filters(epochs)
            features = epochs_filtered.reshape(n_epochs, -1)
        else:
            features = epochs.reshape(n_epochs, -1)

        # Predict
        predictions = self.classifier.predict(features)
        probas = self.classifier.predict_proba(features)
        confidences = probas[:, 1]  # Probability of class 1

        return predictions, confidences

    def _compute_xdawn_filters(self, X: np.ndarray, y: np.ndarray,
                               n_components: int = 4) -> np.ndarray:
        """
        Compute xDAWN spatial filters.
        Simplified implementation using eigenvalue decomposition.

        Args:
            X: Training epochs (n_epochs, n_samples, n_channels)
            y: Labels
            n_components: Number of spatial filters

        Returns:
            Spatial filter matrix (n_channels, n_components)
        """
        # Average target and non-target epochs
        X_target = X[y == 1].mean(axis=0)  # (n_samples, n_channels)
        X_all = X.mean(axis=0)

        # Compute covariance matrices
        C_target = np.cov(X_target.T)  # (n_channels, n_channels)

        # Compute overall covariance
        X_concat = X.reshape(-1, X.shape[2])  # (n_epochs*n_samples, n_channels)
        C_total = np.cov(X_concat.T)

        # Add regularization to avoid singularity
        reg = 0.01 * np.trace(C_total) / C_total.shape[0]
        C_total += reg * np.eye(C_total.shape[0])

        # Generalized eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eig(
                np.linalg.inv(C_total) @ C_target
            )
        except np.linalg.LinAlgError:
            # If decomposition fails, use regularized version
            print("Warning: Singular matrix, using higher regularization")
            reg = 0.1 * np.trace(C_total) / C_total.shape[0]
            C_total += reg * np.eye(C_total.shape[0])
            eigenvalues, eigenvectors = np.linalg.eig(
                np.linalg.inv(C_total) @ C_target
            )

        # Sort by eigenvalues and select top components
        idx = np.argsort(eigenvalues.real)[::-1]
        spatial_filters = eigenvectors[:, idx[:n_components]].real

        return spatial_filters

    def _apply_spatial_filters(self, X: np.ndarray) -> np.ndarray:
        """
        Apply spatial filters to epochs.

        Args:
            X: Epochs (n_epochs, n_samples, n_channels)

        Returns:
            Filtered epochs (n_epochs, n_samples, n_components)
        """
        n_epochs = X.shape[0]
        X_filtered = np.zeros((n_epochs, X.shape[1],
                              self.spatial_filters.shape[1]))

        for i in range(n_epochs):
            X_filtered[i] = X[i] @ self.spatial_filters

        return X_filtered

    def save(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'method': self.method,
            'classifier': self.classifier,
            'spatial_filters': self.spatial_filters,
            'training_metrics': self.training_metrics,
            'n_xdawn_components': self.n_xdawn_components
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.method = model_data['method']
        self.classifier = model_data['classifier']
        self.spatial_filters = model_data['spatial_filters']
        self.training_metrics = model_data['training_metrics']
        self.n_xdawn_components = model_data.get('n_xdawn_components', 4)
        self.is_trained = True

        print(f"Model loaded from {filepath}")
        print(f"Method: {self.method}, CV Accuracy: {self.training_metrics.get('cv_accuracy', 'N/A')}")
