"""
Keyboard-based move selector for demo/testing mode.
Allows selecting moves using arrow keys and spacebar without EEG.
"""

import chess
from typing import List, Optional, Tuple


class KeyboardMoveSelector:
    def __init__(self):
        """Initialize keyboard move selector."""
        # Current cursor position (square index)
        self.cursor_square = chess.E2  # Start at e2 (white pawn)

        # Selection state
        self.selection_step = 0  # 0 = select origin, 1 = select destination
        self.origin_square = None
        self.destination_square = None

        # Available moves for current selection
        self.available_moves = []
        self.current_move_index = 0

        # Board orientation (set externally based on player color)
        self.board_flipped = False

    def move_cursor(self, direction: str):
        """
        Move cursor in a direction.

        Args:
            direction: 'up', 'down', 'left', 'right'
        """
        # Get current file and rank
        file = chess.square_file(self.cursor_square)
        rank = chess.square_rank(self.cursor_square)

        # Adjust direction based on board orientation
        # When board is flipped (Black's perspective), all directions are visually inverted
        if self.board_flipped:
            # Invert all directions for flipped board
            if direction == 'up':
                # Visual up = rank decreases when flipped
                if rank > 0:
                    rank -= 1
            elif direction == 'down':
                # Visual down = rank increases when flipped
                if rank < 7:
                    rank += 1
            elif direction == 'left':
                # Visual left = file increases when flipped
                if file < 7:
                    file += 1
            elif direction == 'right':
                # Visual right = file decreases when flipped
                if file > 0:
                    file -= 1
        else:
            # Normal orientation (White's perspective)
            if direction == 'up' and rank < 7:
                rank += 1
            elif direction == 'down' and rank > 0:
                rank -= 1
            elif direction == 'right' and file < 7:
                file += 1
            elif direction == 'left' and file > 0:
                file -= 1

        # Update cursor square
        self.cursor_square = chess.square(file, rank)

    def select_current_square(self, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
        """
        Select the current square.

        Args:
            legal_moves: List of legal moves in current position

        Returns:
            Complete move if both squares selected, None otherwise
        """
        if self.selection_step == 0:
            # Step 1: Select origin square
            # Filter legal moves that start from this square
            moves_from_square = [m for m in legal_moves
                                if m.from_square == self.cursor_square]

            if moves_from_square:
                self.origin_square = self.cursor_square
                self.available_moves = moves_from_square
                self.current_move_index = 0
                self.selection_step = 1

                # Move cursor to first possible destination
                # Player must confirm with SPACE even if only one move
                self.cursor_square = self.available_moves[0].to_square

            return None

        else:  # selection_step == 1
            # Step 2: Select destination square
            # Find move that goes to current cursor position
            selected_move = None
            for move in self.available_moves:
                if move.to_square == self.cursor_square:
                    selected_move = move
                    break

            # Reset selection state
            self.reset()

            return selected_move

    def cycle_legal_moves(self, direction: int = 1):
        """
        Cycle through legal moves from selected origin.

        Args:
            direction: 1 for next, -1 for previous
        """
        if self.selection_step == 1 and self.available_moves:
            self.current_move_index = (self.current_move_index + direction) % len(self.available_moves)
            self.cursor_square = self.available_moves[self.current_move_index].to_square

    def cancel_selection(self):
        """Cancel current selection and go back to step 1."""
        if self.selection_step == 1:
            self.cursor_square = self.origin_square
            self.selection_step = 0
            self.origin_square = None
            self.available_moves = []
            self.current_move_index = 0

    def reset(self):
        """Reset to initial state."""
        self.selection_step = 0
        self.origin_square = None
        self.destination_square = None
        self.available_moves = []
        self.current_move_index = 0

    def get_highlighted_squares(self) -> dict:
        """
        Get squares to highlight in the UI.

        Returns:
            Dictionary with 'cursor', 'origin', and 'legal_destinations'
        """
        result = {
            'cursor': self.cursor_square,
            'origin': self.origin_square,
            'legal_destinations': []
        }

        if self.selection_step == 1:
            # Show all legal destination squares
            result['legal_destinations'] = [m.to_square for m in self.available_moves]

        return result

    def get_status_text(self) -> str:
        """
        Get status text for UI display.

        Returns:
            Status message string
        """
        square_name = chess.square_name(self.cursor_square)

        if self.selection_step == 0:
            return f"Select origin square (current: {square_name}) - Arrow keys to move, SPACE to select"
        else:
            origin_name = chess.square_name(self.origin_square)
            n_moves = len(self.available_moves)
            current_idx = self.current_move_index + 1

            if n_moves == 1:
                return (f"Selected: {origin_name} → Only move to {square_name} - "
                       f"SPACE to confirm, C/ESC to cancel")
            else:
                return (f"Selected: {origin_name} → Move {current_idx}/{n_moves} to {square_name} - "
                       f"Arrow keys to cycle moves, SPACE to confirm, C/ESC to cancel")
