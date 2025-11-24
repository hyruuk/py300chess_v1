"""
P300-based move selection using row/column flashing paradigm.
"""

import chess
import numpy as np
from typing import List, Dict, Tuple, Optional


class P300MoveSelector:
    def __init__(self, flash_duration: float, isi: float, n_repetitions: int):
        """
        Initialize move selector.

        Args:
            flash_duration: Flash duration in seconds
            isi: Inter-stimulus interval in seconds
            n_repetitions: Number of flash cycles for selection
        """
        self.flash_duration = flash_duration
        self.isi = isi
        self.n_repetitions = n_repetitions

    def create_flash_sequence(self, selection_mode: str = 'square') -> List[Dict]:
        """
        Create row/column flash sequence for board squares.

        Args:
            selection_mode: 'square' for selecting a square, 'piece' for piece selection

        Returns:
            List of flash events with metadata
        """
        # Create groups for rows (0-7, or 1-8 in chess notation) and columns (0-7, or a-h)
        rows = list(range(8))
        cols = list(range(8))

        flash_sequence = []

        for rep in range(self.n_repetitions):
            # Create stimuli for this repetition
            stimuli = []

            # Add row flashes (rank 1-8)
            for row in rows:
                stimuli.append({
                    'type': 'row',
                    'index': row,
                    'rank': row + 1,  # Chess rank (1-8)
                    'squares': [chess.square(col, row) for col in range(8)]
                })

            # Add column flashes (file a-h)
            for col in cols:
                stimuli.append({
                    'type': 'col',
                    'index': col,
                    'file': chr(ord('a') + col),  # Chess file (a-h)
                    'squares': [chess.square(col, row) for row in range(8)]
                })

            # Randomize stimuli order for this repetition
            np.random.shuffle(stimuli)

            # Add repetition index to each stimulus
            for stim in stimuli:
                stim['repetition'] = rep

            flash_sequence.extend(stimuli)

        return flash_sequence

    def determine_selected_square(self, flash_sequence: List[Dict],
                                   p300_scores: List[float]) -> Optional[Tuple[int, Dict]]:
        """
        Determine which square was selected based on P300 scores.

        Args:
            flash_sequence: The flash sequence that was presented
            p300_scores: P300 detection scores for each flash

        Returns:
            (selected_square, confidence_info) or None if cannot determine
        """
        if len(flash_sequence) != len(p300_scores):
            raise ValueError("Flash sequence and scores length mismatch")

        # Aggregate scores by row and column
        row_scores = np.zeros(8)
        col_scores = np.zeros(8)
        row_counts = np.zeros(8)
        col_counts = np.zeros(8)

        for flash, score in zip(flash_sequence, p300_scores):
            if flash['type'] == 'row':
                row_scores[flash['index']] += score
                row_counts[flash['index']] += 1
            elif flash['type'] == 'col':
                col_scores[flash['index']] += score
                col_counts[flash['index']] += 1

        # Average scores
        row_scores = row_scores / np.maximum(row_counts, 1)
        col_scores = col_scores / np.maximum(col_counts, 1)

        # Find best row and column
        best_row = np.argmax(row_scores)
        best_col = np.argmax(col_scores)

        # Determine selected square
        selected_square = chess.square(best_col, best_row)

        # Confidence information
        confidence_info = {
            'row_scores': row_scores.tolist(),
            'col_scores': col_scores.tolist(),
            'best_row': int(best_row),
            'best_col': int(best_col),
            'row_confidence': float(row_scores[best_row]),
            'col_confidence': float(col_scores[best_col]),
            'combined_confidence': float(row_scores[best_row] * col_scores[best_col]),
            'square_name': chess.square_name(selected_square)
        }

        return selected_square, confidence_info

    def create_move_selection_sequence(self, legal_moves: List[chess.Move],
                                       mode: str = 'two_step') -> Dict:
        """
        Create a selection sequence for choosing a move.

        Args:
            legal_moves: List of legal moves
            mode: 'two_step' (select origin then destination) or 'direct'

        Returns:
            Dictionary with selection configuration
        """
        if mode == 'two_step':
            # Step 1: Select origin square
            origin_squares = list(set([move.from_square for move in legal_moves]))

            return {
                'mode': 'two_step',
                'step': 1,
                'step_name': 'select_origin',
                'origin_squares': origin_squares,
                'legal_moves': legal_moves
            }

        elif mode == 'direct':
            # Direct selection: flash all legal moves
            # Group by destination square
            move_by_dest = {}
            for move in legal_moves:
                dest = move.to_square
                if dest not in move_by_dest:
                    move_by_dest[dest] = []
                move_by_dest[dest].append(move)

            return {
                'mode': 'direct',
                'move_by_dest': move_by_dest,
                'legal_moves': legal_moves
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def filter_moves_by_square(self, legal_moves: List[chess.Move],
                               square: int, filter_type: str = 'origin') -> List[chess.Move]:
        """
        Filter legal moves by origin or destination square.

        Args:
            legal_moves: List of legal moves
            square: Square index to filter by
            filter_type: 'origin' or 'destination'

        Returns:
            Filtered list of moves
        """
        if filter_type == 'origin':
            return [move for move in legal_moves if move.from_square == square]
        elif filter_type == 'destination':
            return [move for move in legal_moves if move.to_square == square]
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

    def select_move_two_step(self, legal_moves: List[chess.Move],
                            origin_square: int, dest_square: int) -> Optional[chess.Move]:
        """
        Select a move from origin and destination squares.
        Handles promotion disambiguation if needed.

        Args:
            legal_moves: List of legal moves
            origin_square: Origin square index
            dest_square: Destination square index

        Returns:
            Selected move or None if ambiguous/invalid
        """
        # Filter moves matching origin and destination
        matching_moves = [
            move for move in legal_moves
            if move.from_square == origin_square and move.to_square == dest_square
        ]

        if len(matching_moves) == 0:
            return None
        elif len(matching_moves) == 1:
            return matching_moves[0]
        else:
            # Multiple moves (e.g., pawn promotion)
            # Default to queen promotion
            for move in matching_moves:
                if move.promotion == chess.QUEEN:
                    return move
            # If no queen promotion, return first
            return matching_moves[0]

    def get_flash_timing(self) -> Dict[str, float]:
        """
        Get timing parameters for flashing.

        Returns:
            Dictionary with timing information
        """
        total_time = self.n_repetitions * 16 * (self.flash_duration + self.isi)  # 8 rows + 8 cols = 16

        return {
            'flash_duration': self.flash_duration,
            'isi': self.isi,
            'n_repetitions': self.n_repetitions,
            'flashes_per_repetition': 16,
            'total_flashes': 16 * self.n_repetitions,
            'estimated_time': total_time
        }
