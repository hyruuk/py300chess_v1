"""
Chess game state management using python-chess library.
"""

import chess
from typing import List, Optional, Tuple, Dict


class ChessGameManager:
    def __init__(self):
        """Initialize chess game."""
        self.board = chess.Board()
        self.move_history = []

    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves in current position."""
        return list(self.board.legal_moves)

    def get_legal_moves_grouped(self) -> Dict[str, any]:
        """
        Get legal moves grouped by origin square and destination.
        Useful for row/column flashing paradigm.

        Returns:
            Dictionary with move groupings
        """
        moves = self.get_legal_moves()

        # Group by origin square
        by_origin = {}
        for move in moves:
            origin = chess.square_name(move.from_square)
            if origin not in by_origin:
                by_origin[origin] = []
            by_origin[origin].append(move)

        # Group by destination square
        by_dest = {}
        for move in moves:
            dest = chess.square_name(move.to_square)
            if dest not in by_dest:
                by_dest[dest] = []
            by_dest[dest].append(move)

        return {
            'by_origin': by_origin,
            'by_destination': by_dest,
            'all_moves': moves,
            'n_moves': len(moves)
        }

    def make_move(self, move: chess.Move) -> bool:
        """
        Make a move if legal.

        Args:
            move: Chess move to make

        Returns:
            True if move was made, False if illegal
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            return True
        return False

    def make_move_uci(self, uci_string: str) -> bool:
        """
        Make a move from UCI string (e.g., 'e2e4').

        Args:
            uci_string: UCI format move string

        Returns:
            True if move was made, False if illegal
        """
        try:
            move = chess.Move.from_uci(uci_string)
            return self.make_move(move)
        except ValueError:
            return False

    def undo_move(self) -> Optional[chess.Move]:
        """
        Undo last move.

        Returns:
            The undone move, or None if no moves to undo
        """
        if self.board.move_stack:
            move = self.board.pop()
            self.move_history.pop()
            return move
        return None

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()

    def get_result(self) -> Optional[str]:
        """
        Get game result if game is over.

        Returns:
            Result string ('1-0', '0-1', '1/2-1/2') or None
        """
        if self.is_game_over():
            return self.board.result()
        return None

    def get_game_status(self) -> Dict[str, any]:
        """
        Get comprehensive game status.

        Returns:
            Dictionary with game status information
        """
        status = {
            'fen': self.board.fen(),
            'turn': 'white' if self.board.turn == chess.WHITE else 'black',
            'move_number': self.board.fullmove_number,
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_game_over': self.is_game_over(),
            'result': self.get_result(),
            'n_legal_moves': len(list(self.board.legal_moves))
        }
        return status

    def get_fen(self) -> str:
        """Get current position in FEN notation."""
        return self.board.fen()

    def set_fen(self, fen: str) -> bool:
        """
        Set position from FEN notation.

        Args:
            fen: FEN string

        Returns:
            True if successful, False if invalid FEN
        """
        try:
            self.board.set_fen(fen)
            self.move_history = []
            return True
        except ValueError:
            return False

    def reset(self):
        """Reset to starting position."""
        self.board.reset()
        self.move_history = []

    def get_piece_at(self, square: int) -> Optional[chess.Piece]:
        """
        Get piece at a square.

        Args:
            square: Square index (0-63)

        Returns:
            Piece or None
        """
        return self.board.piece_at(square)

    def get_piece_at_square_name(self, square_name: str) -> Optional[chess.Piece]:
        """
        Get piece at a square by name.

        Args:
            square_name: Square name (e.g., 'e4')

        Returns:
            Piece or None
        """
        try:
            square = chess.parse_square(square_name)
            return self.board.piece_at(square)
        except ValueError:
            return None

    def get_attacked_squares(self, color: chess.Color) -> List[int]:
        """
        Get all squares attacked by a color.

        Args:
            color: chess.WHITE or chess.BLACK

        Returns:
            List of attacked square indices
        """
        attacked = []
        for square in chess.SQUARES:
            if self.board.is_attacked_by(color, square):
                attacked.append(square)
        return attacked

    def get_pgn(self) -> str:
        """
        Get game in PGN notation.

        Returns:
            PGN string
        """
        import io
        from chess import pgn

        game = pgn.Game()
        node = game

        # Replay moves on a temporary board
        temp_board = chess.Board()
        for move in self.move_history:
            node = node.add_variation(move)
            temp_board.push(move)

        # Convert to string
        exporter = pgn.StringExporter(headers=True, variations=True, comments=True)
        return game.accept(exporter)
