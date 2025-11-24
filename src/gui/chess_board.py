"""
Chess board visualization with move highlighting and flashing.
"""

import pygame
import chess
from typing import List, Set, Optional, Dict


class ChessBoardRenderer:
    def __init__(self, board_size: int, position: tuple):
        """
        Initialize chess board renderer.

        Args:
            board_size: Size of board in pixels
            position: (x, y) position of top-left corner
        """
        self.board_size = board_size
        self.position = position
        self.square_size = board_size // 8

        # Colors
        self.light_square = (240, 217, 181)
        self.dark_square = (181, 136, 99)
        self.flash_color = (255, 255, 100)
        self.highlight_color = (130, 151, 105)
        self.last_move_color = (205, 210, 106)
        self.cursor_color = (186, 202, 68)  # Light green for selected square
        self.legal_move_color = (100, 130, 64)  # Dark green for possible moves

        # Current highlights
        self.highlighted_squares = set()
        self.flashing_squares = set()
        self.last_move = None
        self.cursor_square = None
        self.origin_square = None
        self.legal_move_squares = set()

        # Font for coordinate labels
        self.font = pygame.font.Font(None, 20)

        # Load piece images
        self.piece_images = self._load_piece_images()

    def draw(self, surface: pygame.Surface, board: chess.Board,
             show_coordinates: bool = True):
        """
        Draw the chess board and pieces.

        Args:
            surface: Pygame surface to draw on
            board: Current board state
            show_coordinates: Whether to show rank/file labels
        """
        # Draw squares
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)  # Flip vertically for display

                # Determine base color
                is_light = (row + col) % 2 == 0
                color = self.light_square if is_light else self.dark_square

                # Check if last move
                if self.last_move and square in [self.last_move.from_square,
                                                  self.last_move.to_square]:
                    color = self.last_move_color

                # Check if highlighted
                if square in self.highlighted_squares:
                    color = self.highlight_color

                # Draw square
                rect = pygame.Rect(
                    self.position[0] + col * self.square_size,
                    self.position[1] + row * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(surface, color, rect)

                # Flash effect (overlay)
                if square in self.flashing_squares:
                    flash_surface = pygame.Surface(
                        (self.square_size, self.square_size),
                        pygame.SRCALPHA
                    )
                    flash_surface.fill((*self.flash_color, 180))
                    surface.blit(flash_surface, rect.topleft)

                # Draw piece
                piece = board.piece_at(square)
                if piece:
                    self._draw_piece(surface, piece, rect)

        # Draw visual markers for selected and legal move squares
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                rect = pygame.Rect(
                    self.position[0] + col * self.square_size,
                    self.position[1] + row * self.square_size,
                    self.square_size,
                    self.square_size
                )

                # Draw marker for legal move destinations (small circles)
                if square in self.legal_move_squares:
                    self._draw_legal_move_marker(surface, rect)

                # Draw marker for origin piece (the piece that was selected)
                if square == self.origin_square and square != self.cursor_square:
                    self._draw_origin_marker(surface, rect)

                # Draw marker for cursor/selected square (border outline)
                if square == self.cursor_square:
                    self._draw_cursor_marker(surface, rect)

        # Draw coordinate labels
        if show_coordinates:
            self._draw_coordinates(surface)

        # Draw border
        border_rect = pygame.Rect(
            self.position[0],
            self.position[1],
            self.board_size,
            self.board_size
        )
        pygame.draw.rect(surface, (60, 60, 60), border_rect, 3)

    def _load_piece_images(self) -> Dict:
        """
        Load chess piece images from the images/ folder.

        Returns:
            Dictionary mapping (color, piece_type) to pygame Surface
        """
        import os

        piece_images = {}

        # Mapping of piece types to file name characters
        piece_chars = {
            chess.PAWN: 'p',
            chess.KNIGHT: 'n',
            chess.BISHOP: 'b',
            chess.ROOK: 'r',
            chess.QUEEN: 'q',
            chess.KING: 'k'
        }

        # Load all piece images
        for piece_type, char in piece_chars.items():
            for color, color_char in [(chess.WHITE, 'w'), (chess.BLACK, 'b')]:
                filename = f"{color_char}{char}.png"
                filepath = os.path.join("images", filename)

                if os.path.exists(filepath):
                    # Load and scale image to fit square
                    image = pygame.image.load(filepath)
                    # Scale to 90% of square size for some padding
                    scaled_size = int(self.square_size * 0.9)
                    image = pygame.transform.smoothscale(image, (scaled_size, scaled_size))
                    piece_images[(color, piece_type)] = image
                else:
                    print(f"Warning: Could not find piece image: {filepath}")

        return piece_images

    def _draw_piece(self, surface: pygame.Surface, piece: chess.Piece,
                    rect: pygame.Rect):
        """
        Draw a chess piece using loaded images.

        Args:
            surface: Surface to draw on
            piece: Chess piece
            rect: Square rectangle
        """
        # Get the image for this piece
        piece_key = (piece.color, piece.piece_type)

        if piece_key in self.piece_images:
            # Center the image in the square
            image = self.piece_images[piece_key]
            image_rect = image.get_rect(center=rect.center)
            surface.blit(image, image_rect)
        else:
            # Fallback to drawing a colored circle if image not found
            color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
            pygame.draw.circle(surface, color, rect.center, self.square_size // 4)

    def _draw_legal_move_marker(self, surface: pygame.Surface, rect: pygame.Rect):
        """
        Draw a marker for legal move destination squares.
        Uses filled circles in black.

        Args:
            surface: Surface to draw on
            rect: Square rectangle
        """
        # Draw a semi-transparent filled circle in the center
        marker_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        radius = self.square_size // 5  # Larger circle for better visibility
        color = (0, 0, 0, 200)  # Black with high opacity
        pygame.draw.circle(marker_surface, color, (self.square_size // 2, self.square_size // 2), radius)
        surface.blit(marker_surface, rect.topleft)

    def _draw_cursor_marker(self, surface: pygame.Surface, rect: pygame.Rect):
        """
        Draw a marker for the selected/cursor square.
        Uses a thick border outline in black.

        Args:
            surface: Surface to draw on
            rect: Square rectangle
        """
        # Draw a thick black border around the square
        color = (0, 0, 0)  # Black
        border_width = 5  # Thicker for better visibility

        # Draw the border (inset slightly to avoid overlap with board border)
        inner_rect = pygame.Rect(
            rect.left + 2,
            rect.top + 2,
            rect.width - 4,
            rect.height - 4
        )
        pygame.draw.rect(surface, color, inner_rect, border_width)

    def _draw_origin_marker(self, surface: pygame.Surface, rect: pygame.Rect):
        """
        Draw a marker for the origin piece (the piece that was selected).
        Uses corner brackets to indicate the selected piece.

        Args:
            surface: Surface to draw on
            rect: Square rectangle
        """
        color = (0, 0, 0)  # Black
        thickness = 4
        bracket_length = self.square_size // 4  # Length of each corner bracket

        # Draw L-shaped corners in each corner
        # Top-left corner
        pygame.draw.line(surface, color,
                        (rect.left + 4, rect.top + 4),
                        (rect.left + 4 + bracket_length, rect.top + 4), thickness)
        pygame.draw.line(surface, color,
                        (rect.left + 4, rect.top + 4),
                        (rect.left + 4, rect.top + 4 + bracket_length), thickness)

        # Top-right corner
        pygame.draw.line(surface, color,
                        (rect.right - 4, rect.top + 4),
                        (rect.right - 4 - bracket_length, rect.top + 4), thickness)
        pygame.draw.line(surface, color,
                        (rect.right - 4, rect.top + 4),
                        (rect.right - 4, rect.top + 4 + bracket_length), thickness)

        # Bottom-left corner
        pygame.draw.line(surface, color,
                        (rect.left + 4, rect.bottom - 4),
                        (rect.left + 4 + bracket_length, rect.bottom - 4), thickness)
        pygame.draw.line(surface, color,
                        (rect.left + 4, rect.bottom - 4),
                        (rect.left + 4, rect.bottom - 4 - bracket_length), thickness)

        # Bottom-right corner
        pygame.draw.line(surface, color,
                        (rect.right - 4, rect.bottom - 4),
                        (rect.right - 4 - bracket_length, rect.bottom - 4), thickness)
        pygame.draw.line(surface, color,
                        (rect.right - 4, rect.bottom - 4),
                        (rect.right - 4, rect.bottom - 4 - bracket_length), thickness)

    def _draw_coordinates(self, surface: pygame.Surface):
        """Draw rank and file labels."""
        # Files (a-h)
        for col in range(8):
            file_letter = chr(ord('a') + col)
            text = self.font.render(file_letter, True, (180, 180, 180))
            text_rect = text.get_rect(
                centerx=self.position[0] + col * self.square_size + self.square_size // 2,
                y=self.position[1] + self.board_size + 5
            )
            surface.blit(text, text_rect)

        # Ranks (1-8)
        for row in range(8):
            rank_number = str(8 - row)
            text = self.font.render(rank_number, True, (180, 180, 180))
            text_rect = text.get_rect(
                x=self.position[0] - 20,
                centery=self.position[1] + row * self.square_size + self.square_size // 2
            )
            surface.blit(text, text_rect)

    def set_flashing_squares(self, squares: Set[int]):
        """
        Set which squares should flash.

        Args:
            squares: Set of square indices
        """
        self.flashing_squares = squares

    def clear_flash(self):
        """Clear all flashing."""
        self.flashing_squares.clear()

    def set_highlighted_squares(self, squares: Set[int]):
        """
        Set which squares should be highlighted.

        Args:
            squares: Set of square indices
        """
        self.highlighted_squares = squares

    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_squares.clear()

    def set_last_move(self, move: Optional[chess.Move]):
        """
        Set the last move to highlight.

        Args:
            move: Last move or None
        """
        self.last_move = move

    def square_at_position(self, pos: tuple) -> Optional[int]:
        """
        Get square index at screen position.

        Args:
            pos: (x, y) screen position

        Returns:
            Square index or None if outside board
        """
        x, y = pos
        rel_x = x - self.position[0]
        rel_y = y - self.position[1]

        if 0 <= rel_x < self.board_size and 0 <= rel_y < self.board_size:
            col = rel_x // self.square_size
            row = rel_y // self.square_size
            return chess.square(col, 7 - row)

        return None

    def flash_row(self, row: int):
        """
        Flash all squares in a row (rank).

        Args:
            row: Row index (0-7)
        """
        squares = set([chess.square(col, row) for col in range(8)])
        self.set_flashing_squares(squares)

    def flash_column(self, col: int):
        """
        Flash all squares in a column (file).

        Args:
            col: Column index (0-7)
        """
        squares = set([chess.square(col, row) for row in range(8)])
        self.set_flashing_squares(squares)

    def flash_squares_list(self, square_list: List[int]):
        """
        Flash specific squares.

        Args:
            square_list: List of square indices
        """
        self.set_flashing_squares(set(square_list))

    def get_board_rect(self) -> pygame.Rect:
        """Get the rectangle of the board area."""
        return pygame.Rect(
            self.position[0],
            self.position[1],
            self.board_size,
            self.board_size
        )

    def set_cursor_square(self, square: Optional[int]):
        """
        Set cursor square for keyboard selection mode.

        Args:
            square: Square index or None to clear
        """
        self.cursor_square = square

    def set_legal_move_squares(self, squares: List[int]):
        """
        Set legal move destination squares for keyboard mode.

        Args:
            squares: List of square indices
        """
        self.legal_move_squares = set(squares)

    def set_origin_square(self, square: Optional[int]):
        """
        Set origin square (the piece that was selected).

        Args:
            square: Square index or None to clear
        """
        self.origin_square = square

    def clear_keyboard_highlights(self):
        """Clear all keyboard mode highlights."""
        self.cursor_square = None
        self.origin_square = None
        self.legal_move_squares = set()
