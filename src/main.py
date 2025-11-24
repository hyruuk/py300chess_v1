"""
Main application entry point.
Coordinates all components and manages application flow.
"""

import yaml
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import time
import chess

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acquisition.lsl_client import LSLClient
from acquisition.data_buffer import DataBuffer
from processing.preprocessing import EEGPreprocessor
from processing.feature_extraction import P300FeatureExtractor
from classification.calibration import CalibrationSession
from classification.p300_detector import P300Detector
from game.game_manager import ChessGameManager
from game.move_selector import P300MoveSelector
from game.keyboard_selector import KeyboardMoveSelector
from game.engine_interface import ChessEngineInterface
from gui.main_window import MainWindow
from gui.chess_board import ChessBoardRenderer
from gui.flash_controller import FlashController


class P300ChessApp:
    def __init__(self, config_path: str):
        """
        Initialize application.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self._init_components()

        # Application state
        self.mode = 'menu'  # menu, calibration, game
        self.calibration_data = None

    def _init_components(self):
        """Initialize all application components."""
        print("Initializing P300 Chess BCI Application...")

        # EEG acquisition components
        self.lsl_client = LSLClient(
            stream_name=self.config['eeg']['stream_name'],
            expected_channels=self.config['eeg']['channels'],
            sampling_rate=self.config['eeg']['sampling_rate']
        )

        self.data_buffer = DataBuffer(
            n_channels=len(self.config['eeg']['channels']),
            sampling_rate=self.config['eeg']['sampling_rate'],
            buffer_length=self.config['eeg']['buffer_length']
        )

        # Signal processing
        self.preprocessor = EEGPreprocessor(
            sampling_rate=self.config['eeg']['sampling_rate'],
            bandpass=(self.config['processing']['bandpass_low'],
                     self.config['processing']['bandpass_high']),
            notch_freq=self.config['processing']['notch_freq']
        )

        # Get channel indices for target channels
        target_channel_names = self.config['p300']['target_channels']
        all_channels = self.config['eeg']['channels']
        target_indices = [all_channels.index(ch) for ch in target_channel_names
                         if ch in all_channels]

        self.feature_extractor = P300FeatureExtractor(
            target_channels=target_indices,
            sampling_rate=self.config['eeg']['sampling_rate']
        )

        # P300 detection
        self.p300_detector = P300Detector(
            method=self.config['p300']['classifier'],
            n_xdawn_components=self.config['p300']['n_xdawn_components']
        )

        # Chess components
        self.game_manager = ChessGameManager()
        self.move_selector = P300MoveSelector(
            flash_duration=self.config['chess']['flash_duration'],
            isi=self.config['chess']['isi'],
            n_repetitions=self.config['chess']['n_repetitions']
        )

        self.engine = ChessEngineInterface(
            skill_level=self.config['chess']['engine_skill_level'],
            depth=self.config['chess']['engine_depth']
        )

        # GUI (will be initialized in run())
        self.window = None
        self.board_renderer = None
        self.flash_controller = None

        # Game settings
        self.player_color = chess.WHITE  # Player plays as White by default
        self.game_mode = 'demo'  # 'demo' or 'live'

        print("Components initialized successfully!")

    def run(self):
        """Run the application."""
        print("=" * 60)
        print("P300 Chess BCI Application")
        print("=" * 60)

        # Initialize GUI
        self.window = MainWindow(self.config)

        # Create board renderer
        board_size = self.config['gui']['board_size']
        screen_width, screen_height = self.window.get_screen_size()
        board_x = (screen_width - board_size) // 2
        board_y = (screen_height - board_size) // 2

        self.board_renderer = ChessBoardRenderer(
            board_size=board_size,
            position=(board_x, board_y)
        )

        # Create flash controller
        self.flash_controller = FlashController(
            flash_duration=self.config['chess']['flash_duration'],
            isi=self.config['chess']['isi']
        )

        # Set quit callback
        self.window.on_quit = self._cleanup

        # Show main menu
        self._show_menu()

        # Run GUI loop
        self.window.run()

    def _show_menu(self):
        """Show main menu."""
        print("\nMain Menu")
        self.mode = 'menu'

        # Create menu screen
        class MenuScreen:
            def __init__(self, app):
                self.app = app
                self.selected_option = 0  # 0 = mode, 1 = color, 2 = start
                self.blink_counter = 0

                # Click regions (will be set in draw)
                self.mode_rect = None
                self.color_rect = None
                self.start_rect = None
                self.hover_option = None

            def draw(self, surface):
                import pygame
                # Get screen center and mouse position
                cx, cy = self.app.window.get_screen_center()
                mouse_pos = pygame.mouse.get_pos()

                # Set cursor to hand when hovering over clickable areas
                is_hovering = False

                # Title
                self.app.window.draw_text(
                    "P300 Chess BCI",
                    (cx, cy - 200),
                    font=self.app.window.title_font,
                    center=True,
                    color=(220, 220, 255)
                )

                # Menu box
                menu_y = cy - 80
                option_height = 70

                # Option 1: Mode Selection
                mode_text = "DEMO MODE" if self.app.game_mode == 'demo' else "LIVE MODE (P300)"
                mode_color = (100, 255, 100) if self.app.game_mode == 'demo' else (255, 200, 100)

                # Create clickable region for mode
                self.mode_rect = pygame.Rect(cx - 50, menu_y - 15, 300, 50)
                is_mode_hover = self.mode_rect.collidepoint(mouse_pos)

                # Draw hover background
                if is_mode_hover:
                    pygame.draw.rect(surface, (50, 50, 50), self.mode_rect)
                    pygame.draw.rect(surface, (150, 150, 150), self.mode_rect, 2)

                self.app.window.draw_text(
                    "Mode:",
                    (cx - 200, menu_y),
                    font=self.app.window.large_font,
                    color=(180, 180, 180)
                )
                self.app.window.draw_text(
                    mode_text,
                    (cx + 100, menu_y),
                    font=self.app.window.large_font,
                    center=True,
                    color=mode_color if (self.selected_option == 0 or is_mode_hover) else (150, 150, 150)
                )
                if self.selected_option == 0 or is_mode_hover:
                    self.app.window.draw_text(
                        "< Click or Press 1/2 >",
                        (cx + 100, menu_y + 30),
                        font=self.app.window.font,
                        center=True,
                        color=(150, 200, 255)
                    )

                # Option 2: Color Selection
                menu_y += option_height
                color_text = "WHITE" if self.app.player_color == chess.WHITE else "BLACK"
                color_color = (255, 255, 255) if self.app.player_color == chess.WHITE else (150, 150, 150)

                # Create clickable region for color
                self.color_rect = pygame.Rect(cx - 50, menu_y - 15, 300, 50)
                is_color_hover = self.color_rect.collidepoint(mouse_pos)

                # Draw hover background
                if is_color_hover:
                    pygame.draw.rect(surface, (50, 50, 50), self.color_rect)
                    pygame.draw.rect(surface, (150, 150, 150), self.color_rect, 2)

                self.app.window.draw_text(
                    "Play as:",
                    (cx - 200, menu_y),
                    font=self.app.window.large_font,
                    color=(180, 180, 180)
                )
                self.app.window.draw_text(
                    color_text,
                    (cx + 100, menu_y),
                    font=self.app.window.large_font,
                    center=True,
                    color=color_color if (self.selected_option == 1 or is_color_hover) else (120, 120, 120)
                )
                if self.selected_option == 1 or is_color_hover:
                    self.app.window.draw_text(
                        "< Click or Press W/B >",
                        (cx + 100, menu_y + 30),
                        font=self.app.window.font,
                        center=True,
                        color=(150, 200, 255)
                    )

                # Start Game button
                menu_y += option_height + 40

                # Create clickable region for start button
                self.start_rect = pygame.Rect(cx - 150, menu_y - 10, 300, 50)
                is_start_hover = self.start_rect.collidepoint(mouse_pos)

                # Blinking effect for START
                self.blink_counter = (self.blink_counter + 1) % 60
                start_alpha = 255 if self.blink_counter < 30 or self.selected_option != 2 else 150
                start_color = (100, 255, 100, start_alpha) if (self.selected_option == 2 or is_start_hover) else (200, 200, 200)

                pygame.draw.rect(surface, (60, 60, 60), self.start_rect)
                if is_start_hover:
                    pygame.draw.rect(surface, start_color[:3], self.start_rect, 5)
                else:
                    pygame.draw.rect(surface, start_color[:3], self.start_rect, 3)

                self.app.window.draw_text(
                    "START GAME",
                    (cx, menu_y + 15),
                    font=self.app.window.large_font,
                    center=True,
                    color=start_color[:3]
                )

                # Instructions at bottom
                instructions_y = self.app.window.get_screen_size()[1] - 120

                instructions = [
                    "Click options to change  |  Arrow Keys: Navigate",
                    "1/2: Mode  |  W/B: Color  |  ENTER/SPACE: Start",
                    "C: Calibration  |  ESC: Quit"
                ]

                for i, instruction in enumerate(instructions):
                    self.app.window.draw_text(
                        instruction,
                        (cx, instructions_y + i * 30),
                        font=self.app.window.font,
                        center=True,
                        color=(150, 150, 150)
                    )

                # Set mouse cursor based on hover state
                if is_mode_hover or is_color_hover or is_start_hover:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                else:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

            def handle_event(self, event):
                import pygame

                # Mouse click handling
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pos = event.pos

                        # Check mode selection click
                        if self.mode_rect and self.mode_rect.collidepoint(mouse_pos):
                            # Toggle mode
                            self.app.game_mode = 'live' if self.app.game_mode == 'demo' else 'demo'
                            print(f"Mode: {self.app.game_mode.upper()}")

                        # Check color selection click
                        elif self.color_rect and self.color_rect.collidepoint(mouse_pos):
                            # Toggle color
                            self.app.player_color = chess.BLACK if self.app.player_color == chess.WHITE else chess.WHITE
                            print(f"Playing as: {'WHITE' if self.app.player_color == chess.WHITE else 'BLACK'}")

                        # Check start button click
                        elif self.start_rect and self.start_rect.collidepoint(mouse_pos):
                            self.app._start_game()

                # Keyboard handling
                elif event.type == pygame.KEYDOWN:
                    # Quit application from menu
                    if event.key == pygame.K_ESCAPE:
                        print("Quitting application...")
                        self.app.window.running = False

                    # Mode selection
                    elif event.key == pygame.K_1:
                        self.app.game_mode = 'demo'
                        print("Mode: DEMO (Keyboard)")
                    elif event.key == pygame.K_2:
                        self.app.game_mode = 'live'
                        print("Mode: LIVE (P300)")

                    # Color selection
                    elif event.key == pygame.K_w:
                        self.app.player_color = chess.WHITE
                        print("Playing as: WHITE")
                    elif event.key == pygame.K_b:
                        self.app.player_color = chess.BLACK
                        print("Playing as: BLACK")

                    # Navigation
                    elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        self.selected_option = (self.selected_option + 1) % 3

                    # Start game
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        self.app._start_game()

                    # Calibration
                    elif event.key == pygame.K_c:
                        self.app._start_calibration()

            def update(self):
                pass

        self.window.set_screen(MenuScreen(self))

    def _start_calibration(self):
        """Start calibration procedure."""
        print("\n" + "=" * 60)
        print("Starting Calibration")
        print("=" * 60)

        # Try to connect to LSL stream
        if not self._connect_lsl():
            self.window.show_message(
                "Connection Failed",
                "Could not connect to EEG stream. Check LSL connection."
            )
            self._show_menu()
            return

        # Create calibration session
        calibration = CalibrationSession(
            n_targets=8,  # 8 rows or columns
            n_trials_per_target=self.config['calibration']['n_trials_per_stimulus'],
            n_nontargets_per_target=self.config['calibration']['n_nontarget_per_target'],
            flash_duration=self.config['calibration']['flash_duration'],
            isi=self.config['calibration']['isi']
        )

        # Run calibration (simplified version)
        print("Calibration session created")
        print(f"Total trials: {len(calibration.trials)}")

        # For now, show message and return to menu
        # Full implementation would present visual stimuli and collect data
        self.window.show_message(
            "Calibration",
            "Calibration mode - Full implementation requires visual stimulus presentation",
            duration=3.0
        )

        self._show_menu()

    def _start_game(self):
        """Start chess game."""
        print("\n" + "=" * 60)
        print("Starting Game")
        print("=" * 60)
        print(f"Mode: {self.game_mode.upper()}")
        print(f"Playing as: {'WHITE' if self.player_color == chess.WHITE else 'BLACK'}")

        # Check if live mode requires calibration
        if self.game_mode == 'live' and not self.p300_detector.is_trained:
            print("Error: Live mode requires calibration first")
            self._show_menu()
            return

        # Start chess engine if needed
        if self.config.get('demo', {}).get('auto_play_engine', True):
            try:
                self.engine.start()
                print("Chess engine started successfully")
            except Exception as e:
                print(f"Warning: Could not start chess engine: {e}")
                print("Game will continue without engine opponent")

        # Reset game
        self.game_manager.reset()

        # Create appropriate game screen based on mode
        if self.game_mode == 'demo':
            print("Starting in KEYBOARD DEMO MODE")
            self.window.set_screen(self._create_keyboard_game_screen())
        else:
            print("Starting in P300 MODE")
            self.window.set_screen(self._create_p300_game_screen())

    def _create_keyboard_game_screen(self):
        """Create keyboard-controlled game screen for demo mode."""
        import pygame

        class KeyboardGameScreen:
            def __init__(self, app):
                self.app = app
                self.keyboard_selector = KeyboardMoveSelector()
                self.waiting_for_engine = False
                self.game_start_time = time.time()
                self.forfeit_button_rect = None

                # If player is Black, engine makes first move
                if self.app.player_color == chess.BLACK:
                    if self.app.engine.is_running:
                        self.waiting_for_engine = True
                        print("Player is Black - waiting for engine's first move")
                    # Set cursor to a Black piece
                    self.keyboard_selector.cursor_square = chess.E7
                else:
                    # Set cursor to a White piece
                    self.keyboard_selector.cursor_square = chess.E2

            def draw(self, surface):
                import pygame
                # Reset cursor to arrow (in case it was hand from menu)
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

                # Set board orientation based on player color
                self.app.board_renderer.flipped = (self.app.player_color == chess.BLACK)
                self.keyboard_selector.board_flipped = (self.app.player_color == chess.BLACK)

                # Update board highlights
                highlights = self.keyboard_selector.get_highlighted_squares()
                self.app.board_renderer.set_cursor_square(highlights['cursor'])
                self.app.board_renderer.set_legal_move_squares(highlights['legal_destinations'])
                self.app.board_renderer.set_origin_square(highlights['origin'])

                # Draw chess board
                self.app.board_renderer.draw(
                    surface,
                    self.app.game_manager.board
                )

                # Draw game status
                status = self.app.game_manager.get_game_status()

                # Highlight whose turn it is
                turn_color = (255, 255, 255) if status['turn'] == 'white' else (150, 150, 150)
                status_text = f"Turn: {status['turn'].upper()}  |  Move: {status['move_number']}"

                if status['is_check']:
                    status_text += "  |  CHECK!"

                self.app.window.draw_text(
                    status_text,
                    (20, 20),
                    font=self.app.window.large_font,
                    color=turn_color
                )

                # Draw player color and engine info
                player_color_text = "You: WHITE" if self.app.player_color == chess.WHITE else "You: BLACK"
                player_text_color = (255, 255, 255) if self.app.player_color == chess.WHITE else (150, 150, 150)

                engine_level = self.app.config['chess']['engine_skill_level']
                engine_status = "Manual Play" if not self.app.config.get('demo', {}).get('auto_play_engine', True) else f"vs Engine Lvl {engine_level}"

                info_text = f"{player_color_text}  |  {engine_status}"
                self.app.window.draw_text(
                    info_text,
                    (self.app.window.get_screen_size()[0] - 350, 20),
                    font=self.app.window.large_font,
                    color=player_text_color
                )

                # Draw keyboard selector status
                selector_status = self.keyboard_selector.get_status_text()
                self.app.window.draw_text(
                    selector_status,
                    (20, 60),
                    font=self.app.window.font
                )

                # Instructions
                instructions = "Arrow keys: Move  |  SPACE: Select  |  C/BACKSPACE: Cancel  |  ESC: Forfeit  |  +/-: Difficulty"
                self.app.window.draw_text(
                    instructions,
                    (20, self.app.window.get_screen_size()[1] - 30)
                )

                # Forfeit button (only show if game not over)
                if not status['is_game_over']:
                    mouse_pos = pygame.mouse.get_pos()
                    button_width = 120
                    button_height = 40
                    button_x = self.app.window.get_screen_size()[0] - button_width - 20
                    button_y = self.app.window.get_screen_size()[1] - button_height - 20

                    self.forfeit_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
                    is_hover = self.forfeit_button_rect.collidepoint(mouse_pos)

                    # Draw button
                    button_color = (100, 30, 30) if is_hover else (70, 20, 20)
                    border_color = (200, 60, 60) if is_hover else (150, 50, 50)

                    pygame.draw.rect(surface, button_color, self.forfeit_button_rect)
                    pygame.draw.rect(surface, border_color, self.forfeit_button_rect, 3 if is_hover else 2)

                    self.app.window.draw_text(
                        "FORFEIT",
                        (button_x + button_width // 2, button_y + 8),
                        font=self.app.window.font,
                        center=True,
                        color=(255, 100, 100) if is_hover else (200, 80, 80)
                    )
                    # Small ESC hint
                    small_font = pygame.font.Font(None, 16)
                    self.app.window.draw_text(
                        "(ESC)",
                        (button_x + button_width // 2, button_y + 24),
                        font=small_font,
                        center=True,
                        color=(150, 70, 70) if not is_hover else (200, 90, 90)
                    )

                    # Set cursor to hand when hovering
                    if is_hover:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    else:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

                # Game over overlay
                if status['is_game_over']:
                    self._draw_game_over_screen(surface, status)

            def handle_event(self, event):
                import pygame

                # Mouse click handling
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Check forfeit button click
                        if self.forfeit_button_rect and self.forfeit_button_rect.collidepoint(event.pos):
                            if not self.app.game_manager.is_game_over():
                                self._forfeit_game()
                            return

                if event.type == pygame.KEYDOWN:
                    # If game is over, only allow ESC to return to menu
                    if self.app.game_manager.is_game_over():
                        if event.key == pygame.K_ESCAPE:
                            self.app._show_menu()
                        return

                    if event.key == pygame.K_ESCAPE:
                        if self.keyboard_selector.selection_step == 1:
                            # Cancel selection
                            self.keyboard_selector.cancel_selection()
                            print("Selection cancelled")
                        else:
                            # Forfeit the game
                            self._forfeit_game()

                    elif event.key == pygame.K_BACKSPACE or event.key == pygame.K_c:
                        # Additional cancel keys
                        if self.keyboard_selector.selection_step == 1:
                            self.keyboard_selector.cancel_selection()
                            print("Selection cancelled")

                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                        # Increase engine difficulty
                        current_level = self.app.config['chess']['engine_skill_level']
                        if current_level < 20:
                            new_level = current_level + 1
                            self.app.config['chess']['engine_skill_level'] = new_level
                            if self.app.engine.is_running:
                                self.app.engine.set_skill_level(new_level)
                            print(f"Engine difficulty increased to {new_level}")

                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        # Decrease engine difficulty
                        current_level = self.app.config['chess']['engine_skill_level']
                        if current_level > 0:
                            new_level = current_level - 1
                            self.app.config['chess']['engine_skill_level'] = new_level
                            if self.app.engine.is_running:
                                self.app.engine.set_skill_level(new_level)
                            print(f"Engine difficulty decreased to {new_level}")

                    elif event.key == pygame.K_UP:
                        if self.keyboard_selector.selection_step == 1:
                            self.keyboard_selector.cycle_legal_moves(1)
                        else:
                            self.keyboard_selector.move_cursor('up')

                    elif event.key == pygame.K_DOWN:
                        if self.keyboard_selector.selection_step == 1:
                            self.keyboard_selector.cycle_legal_moves(-1)
                        else:
                            self.keyboard_selector.move_cursor('down')

                    elif event.key == pygame.K_LEFT:
                        if self.keyboard_selector.selection_step == 0:
                            self.keyboard_selector.move_cursor('left')

                    elif event.key == pygame.K_RIGHT:
                        if self.keyboard_selector.selection_step == 0:
                            self.keyboard_selector.move_cursor('right')

                    elif event.key == pygame.K_SPACE:
                        # Select current square
                        legal_moves = self.app.game_manager.get_legal_moves()
                        move = self.keyboard_selector.select_current_square(legal_moves)

                        if move:
                            # Execute move
                            print(f"Player move: {move}")
                            self.app.game_manager.make_move(move)
                            self.app.board_renderer.set_last_move(move)

                            # Check if game is over
                            if self.app.game_manager.is_game_over():
                                print(f"Game over: {self.app.game_manager.get_result()}")
                                return

                            # Engine responds if enabled AND it's the engine's turn
                            if self.app.config.get('demo', {}).get('auto_play_engine', True):
                                # Engine plays the opposite color
                                current_turn = self.app.game_manager.board.turn
                                if current_turn != self.app.player_color:
                                    self.waiting_for_engine = True
                                else:
                                    # Still player's turn (shouldn't happen normally)
                                    self._move_cursor_to_current_turn_piece()
                            else:
                                # Manual play mode - move cursor to a piece of the current turn
                                self._move_cursor_to_current_turn_piece()

            def update(self):
                # Make engine move if waiting
                if self.waiting_for_engine:
                    if self.app.engine.is_running:
                        try:
                            engine_move = self.app.engine.get_best_move(
                                self.app.game_manager.board
                            )
                            if engine_move:
                                print(f"Engine move: {engine_move}")
                                self.app.game_manager.make_move(engine_move)
                                self.app.board_renderer.set_last_move(engine_move)
                                # Move cursor to a piece of the current turn
                                self._move_cursor_to_current_turn_piece()
                        except Exception as e:
                            print(f"Engine error: {e}")
                    else:
                        # Engine not running - allow manual play for both sides
                        print("Engine not available - manual play mode")
                        # Move cursor to current turn's pieces
                        self._move_cursor_to_current_turn_piece()

                    self.waiting_for_engine = False

            def _move_cursor_to_current_turn_piece(self):
                """Move cursor to a piece that can move in the current turn."""
                # Get current turn
                current_turn = self.app.game_manager.board.turn

                # Try to find a piece of the current player that has legal moves
                for square in chess.SQUARES:
                    piece = self.app.game_manager.board.piece_at(square)
                    if piece and piece.color == current_turn:
                        # Check if this piece has any legal moves
                        legal_moves = self.app.game_manager.get_legal_moves()
                        if any(move.from_square == square for move in legal_moves):
                            self.keyboard_selector.cursor_square = square
                            return

                # Fallback: just find any piece of the current color
                for square in chess.SQUARES:
                    piece = self.app.game_manager.board.piece_at(square)
                    if piece and piece.color == current_turn:
                        self.keyboard_selector.cursor_square = square
                        return

            def _forfeit_game(self):
                """Forfeit the current game."""
                print("Player forfeited the game")
                # Mark who forfeited in the game manager
                self.app.game_manager.forfeited_by = self.app.player_color
                # Create a dummy game over state by resigning
                self.app.game_manager.board.push(chess.Move.null())  # Illegal move to end game
                # Actually, better to just track it separately
                self.app.game_manager.board.pop()  # Remove that
                # Just track forfeit state
                self.app.game_manager.is_forfeited = True

            def _draw_game_over_screen(self, surface, status):
                """Draw game over overlay with winner and stats."""
                import pygame

                # Semi-transparent overlay
                overlay = pygame.Surface((self.app.window.get_screen_size()[0],
                                         self.app.window.get_screen_size()[1]))
                overlay.set_alpha(200)
                overlay.fill((30, 30, 30))
                surface.blit(overlay, (0, 0))

                # Frame dimensions
                frame_width = 700
                frame_height = 500
                frame_x = (self.app.window.get_screen_size()[0] - frame_width) // 2
                frame_y = (self.app.window.get_screen_size()[1] - frame_height) // 2

                # Draw frame background
                frame_rect = pygame.Rect(frame_x, frame_y, frame_width, frame_height)
                pygame.draw.rect(surface, (50, 50, 50), frame_rect)
                pygame.draw.rect(surface, (200, 200, 200), frame_rect, 4)

                # Determine winner
                result = status['result']
                if result == '1-0':
                    winner_text = "WHITE WINS!"
                    winner_color = (255, 255, 255)
                elif result == '0-1':
                    winner_text = "BLACK WINS!"
                    winner_color = (150, 150, 150)
                else:
                    winner_text = "DRAW!"
                    winner_color = (200, 200, 100)

                # Draw winner text
                self.app.window.draw_text(
                    winner_text,
                    (frame_x + frame_width // 2, frame_y + 50),
                    font=self.app.window.title_font,
                    center=True,
                    color=winner_color
                )

                # Determine reason for game over
                reason = ""
                if self.app.game_manager.is_forfeited:
                    forfeiter = "WHITE" if self.app.game_manager.forfeited_by == chess.WHITE else "BLACK"
                    reason = f"{forfeiter} Forfeited"
                elif status['is_checkmate']:
                    reason = "by Checkmate"
                elif status['is_stalemate']:
                    reason = "by Stalemate"
                else:
                    outcome = self.app.game_manager.board.outcome()
                    if outcome:
                        reason = outcome.termination.name.replace('_', ' ').title()
                    else:
                        reason = ""

                self.app.window.draw_text(
                    reason,
                    (frame_x + frame_width // 2, frame_y + 100),
                    font=self.app.window.large_font,
                    center=True,
                    color=(180, 180, 180)
                )

                # Game statistics
                stats_y = frame_y + 160
                line_height = 35

                # Calculate game duration
                game_duration = int(time.time() - self.game_start_time)
                minutes = game_duration // 60
                seconds = game_duration % 60

                stats = [
                    f"Total Moves: {status['move_number']}",
                    f"Game Duration: {minutes}m {seconds}s",
                ]

                # Count captures
                piece_count = {'white': 0, 'black': 0}
                for square in chess.SQUARES:
                    piece = self.app.game_manager.board.piece_at(square)
                    if piece:
                        if piece.color == chess.WHITE:
                            piece_count['white'] += 1
                        else:
                            piece_count['black'] += 1

                captures_white = 16 - piece_count['black']
                captures_black = 16 - piece_count['white']
                stats.append(f"Pieces Captured: White {captures_white} - Black {captures_black}")

                # Add engine info if available
                if self.app.engine.is_running:
                    stats.append(f"Engine Difficulty: {self.app.config['chess']['engine_skill_level']}/20")

                # Add FEN for advanced users
                stats.append("")
                stats.append(f"FEN: {self.app.game_manager.get_fen()}")

                # Draw stats
                for i, stat in enumerate(stats):
                    self.app.window.draw_text(
                        stat,
                        (frame_x + frame_width // 2, stats_y + i * line_height),
                        font=self.app.window.font,
                        center=True,
                        color=(220, 220, 220)
                    )

                # Instructions to continue
                self.app.window.draw_text(
                    "Press ESC to return to menu",
                    (frame_x + frame_width // 2, frame_y + frame_height - 40),
                    font=self.app.window.large_font,
                    center=True,
                    color=(150, 200, 255)
                )

        return KeyboardGameScreen(self)

    def _create_p300_game_screen(self):
        """Create P300-controlled game screen."""
        import pygame

        class P300GameScreen:
            def __init__(self, app):
                self.app = app

            def draw(self, surface):
                # Draw chess board
                self.app.board_renderer.draw(
                    surface,
                    self.app.game_manager.board
                )

                # Draw game status
                status = self.app.game_manager.get_game_status()
                status_text = f"Turn: {status['turn']}  |  Move: {status['move_number']}"

                if status['is_check']:
                    status_text += "  |  CHECK!"

                self.app.window.draw_text(
                    status_text,
                    (20, 20),
                    font=self.app.window.large_font
                )

                # Instructions
                self.app.window.draw_text(
                    "P300 Mode - M: Menu",
                    (20, self.app.window.get_screen_size()[1] - 30)
                )

            def handle_event(self, event):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.app._show_menu()

            def update(self):
                pass

        return P300GameScreen(self)

    def _connect_lsl(self) -> bool:
        """
        Connect to LSL stream.

        Returns:
            True if successful, False otherwise
        """
        print("Connecting to LSL stream...")
        try:
            if self.lsl_client.connect(timeout=5.0):
                self.lsl_client.start_acquisition(self._on_eeg_sample)
                return True
        except Exception as e:
            print(f"Error connecting to LSL: {e}")

        return False

    def _on_eeg_sample(self, sample, timestamp):
        """Callback for incoming EEG samples."""
        self.data_buffer.add_sample(sample, timestamp)

    def _cleanup(self):
        """Cleanup resources."""
        print("\nShutting down...")

        # Stop EEG acquisition
        if self.lsl_client:
            self.lsl_client.stop_acquisition()

        # Stop chess engine
        if self.engine:
            self.engine.stop()

        print("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='P300 Chess BCI')
    parser.add_argument('--config', type=str,
                       default='config/settings.yaml',
                       help='Path to configuration file')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (no EEG required)')

    args = parser.parse_args()

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print(f"Please create config file or specify correct path")
        return 1

    try:
        # Create and run application
        app = P300ChessApp(args.config)
        app.run()
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
