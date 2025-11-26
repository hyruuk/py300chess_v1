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
import pickle
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acquisition.lsl_client import LSLClient
from acquisition.data_buffer import DataBuffer
from acquisition.simulated_eeg import SimulatedEEGStream
from processing.preprocessing import EEGPreprocessor
from processing.feature_extraction import P300FeatureExtractor
from classification.calibration import CalibrationSession
from classification.p300_detector import P300Detector
from game.game_manager import ChessGameManager
from game.move_selector import P300MoveSelector
from game.keyboard_selector import KeyboardMoveSelector
from game.engine_interface import ChessEngineInterface
from game.live_p300_engine import LiveP300Engine
from gui.main_window import MainWindow
from gui.chess_board import ChessBoardRenderer
from gui.flash_controller import FlashController
from gui.eeg_visualizer import EEGVisualizer


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
        self.is_calibrated = False  # Track calibration status

        # Simulated EEG stream (if using simulated mode)
        self.simulated_stream = None

        # EEG visualizer (optional)
        self.eeg_visualizer = None
        self.show_eeg_viz = False

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

        # Live P300 engine
        self.live_p300_engine = LiveP300Engine(
            data_buffer=self.data_buffer,
            preprocessor=self.preprocessor,
            feature_extractor=self.feature_extractor,
            p300_detector=self.p300_detector,
            use_random_selection=True  # Use random for testing until trained
        )

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
                self.selected_option = 0  # 0 = mode, 1 = color, 2 = stream, 3 = start
                self.blink_counter = 0

                # Click regions (will be set in draw)
                self.mode_rect = None
                self.color_rect = None
                self.stream_rect = None
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

                # Option 3: Stream Type (only show in LIVE mode)
                if self.app.game_mode == 'live':
                    menu_y += option_height
                    use_simulated = self.app.config['eeg'].get('use_simulated', False)
                    stream_text = "SIMULATED" if use_simulated else "REAL EEG"
                    stream_color = (100, 200, 255) if use_simulated else (255, 150, 100)

                    # Create clickable region for stream
                    self.stream_rect = pygame.Rect(cx - 50, menu_y - 15, 300, 50)
                    is_stream_hover = self.stream_rect.collidepoint(mouse_pos)

                    # Draw hover background
                    if is_stream_hover:
                        pygame.draw.rect(surface, (50, 50, 50), self.stream_rect)
                        pygame.draw.rect(surface, (150, 150, 150), self.stream_rect, 2)

                    self.app.window.draw_text(
                        "EEG Stream:",
                        (cx - 200, menu_y),
                        font=self.app.window.large_font,
                        color=(180, 180, 180)
                    )
                    self.app.window.draw_text(
                        stream_text,
                        (cx + 100, menu_y),
                        font=self.app.window.large_font,
                        center=True,
                        color=stream_color if (self.selected_option == 2 or is_stream_hover) else (150, 150, 150)
                    )
                    if self.selected_option == 2 or is_stream_hover:
                        self.app.window.draw_text(
                            "< Click or Press S/R >",
                            (cx + 100, menu_y + 30),
                            font=self.app.window.font,
                            center=True,
                            color=(150, 200, 255)
                        )
                else:
                    self.stream_rect = None

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

                # Calibration status indicator
                if self.app.game_mode == 'live':
                    calib_y = menu_y + option_height
                    calib_status = "CALIBRATED" if self.app.is_calibrated else "NOT CALIBRATED"
                    calib_color = (100, 255, 100) if self.app.is_calibrated else (255, 100, 100)

                    self.app.window.draw_text(
                        "Calibration:",
                        (cx - 200, calib_y),
                        font=self.app.window.large_font,
                        color=(180, 180, 180)
                    )
                    self.app.window.draw_text(
                        calib_status,
                        (cx + 100, calib_y),
                        font=self.app.window.large_font,
                        center=True,
                        color=calib_color
                    )

                # Instructions at bottom
                instructions_y = self.app.window.get_screen_size()[1] - 120

                if self.app.game_mode == 'live':
                    instructions = [
                        "Click options to change  |  Arrow Keys: Navigate",
                        "1/2: Mode  |  W/B: Color  |  S/R: Stream  |  ENTER/SPACE: Start",
                        "C: Calibration  |  ESC: Quit"
                    ]
                else:
                    instructions = [
                        "Click options to change  |  Arrow Keys: Navigate",
                        "1/2: Mode  |  W/B: Color  |  ENTER/SPACE: Start",
                        "ESC: Quit"
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
                is_stream_hover_check = self.stream_rect and self.stream_rect.collidepoint(mouse_pos)
                if is_mode_hover or is_color_hover or is_stream_hover_check or is_start_hover:
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

                        # Check stream selection click (only in live mode)
                        elif self.stream_rect and self.stream_rect.collidepoint(mouse_pos):
                            # Toggle stream type
                            current = self.app.config['eeg'].get('use_simulated', False)
                            self.app.config['eeg']['use_simulated'] = not current
                            stream_type = "SIMULATED" if not current else "REAL"
                            print(f"EEG Stream: {stream_type}")

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

                    # Stream selection (only in live mode)
                    elif event.key == pygame.K_s:
                        if self.app.game_mode == 'live':
                            self.app.config['eeg']['use_simulated'] = True
                            print("EEG Stream: SIMULATED")
                    elif event.key == pygame.K_r:
                        if self.app.game_mode == 'live':
                            self.app.config['eeg']['use_simulated'] = False
                            print("EEG Stream: REAL")

                    # Navigation
                    elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        max_options = 4 if self.app.game_mode == 'live' else 3
                        self.selected_option = (self.selected_option + 1) % max_options

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

        # Create calibration screen
        self.window.set_screen(self._create_calibration_screen())

    def _start_game(self):
        """Start chess game."""
        print("\n" + "=" * 60)
        print("Starting Game")
        print("=" * 60)
        print(f"Mode: {self.game_mode.upper()}")
        print(f"Playing as: {'WHITE' if self.player_color == chess.WHITE else 'BLACK'}")

        # Check if live mode requires calibration
        if self.game_mode == 'live' and not self.is_calibrated:
            print("Error: Live mode requires calibration first!")
            print("Please run calibration (press C) before starting a Live game.")
            # Return to menu without starting game
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
        import random

        class P300GameScreen:
            def __init__(self, app):
                self.app = app
                self.state = 'connecting'  # connecting, keyboard_focus, waiting_player, flashing, processing, done
                self.status_message = "Connecting to EEG stream..."

                # Flash sequence
                self.flash_sequence = []
                self.sequence_results = None

                # Selection state
                self.selection_step = 0  # 0 = origin, 1 = destination
                self.selected_origin = None
                self.legal_moves_from_origin = []

                # Keyboard focus mode
                self.focused_square = chess.E2  # Start at e2
                self.keyboard_mode_enabled = True  # Start in keyboard mode to set focus

                # Connect to LSL in background
                self._start_connection()

            def _start_connection(self):
                """Start connection to LSL stream."""
                # Try to connect
                success = self.app._connect_lsl()

                if success:
                    # Create EEG visualizer component (on the right side of screen)
                    if not self.app.eeg_visualizer:
                        screen_width, screen_height = self.app.window.get_screen_size()
                        board_size = self.app.config['gui']['board_size']

                        # Position visualizer on the right side
                        viz_x = (screen_width + board_size) // 2 + 20
                        viz_y = 50
                        viz_width = screen_width - viz_x - 20
                        viz_height = screen_height - 100

                        self.app.eeg_visualizer = EEGVisualizer(
                            n_channels=len(self.app.config['eeg']['channels']),
                            channel_names=self.app.config['eeg']['channels'],
                            sampling_rate=self.app.config['eeg']['sampling_rate'],
                            position=(viz_x, viz_y),
                            width=viz_width,
                            height=viz_height,
                            window_duration=5.0
                        )
                        print("EEG visualizer created")

                    self.state = 'keyboard_focus'
                    self.status_message = "Use arrow keys to select focus square, then press ENTER to start"
                    # Highlight the initial focused square
                    self.app.board_renderer.set_highlighted_squares({self.focused_square})
                else:
                    self.state = 'error'
                    self.status_message = "Failed to connect to EEG stream"

            def _start_flash_sequence(self):
                """Start a row/column flash sequence."""
                self.state = 'flashing'
                self.status_message = "Watch the flashing squares..."

                # Use focused square as target for simulated stream
                if self.app.simulated_stream:
                    target_square = self.focused_square
                    target_row = chess.square_rank(target_square)
                    target_col = chess.square_file(target_square)

                    # Set target for simulated stream
                    self.app.simulated_stream.set_target(target_row, target_col)
                    self.target_square = target_square

                    print(f"SimulatedEEG target: {chess.square_name(target_square)} "
                          f"(row={target_row}, col={target_col})")

                # Create flash sequence (only flash legal origin squares)
                legal_moves = self.app.game_manager.get_legal_moves()
                legal_origin_squares = list(set(move.from_square for move in legal_moves))
                self.flash_sequence = self.app.move_selector.create_flash_sequence(
                    legal_squares=legal_origin_squares
                )

                # Start flash controller
                self.app.flash_controller.start_sequence(
                    flash_sequence=self.flash_sequence,
                    on_flash_start=self._on_flash_start,
                    on_flash_end=self._on_flash_end,
                    on_sequence_complete=self._on_sequence_complete
                )

                # Start live P300 engine
                self.app.live_p300_engine.start_sequence(
                    sequence_id=f"selection_{self.selection_step}",
                    n_flashes=len(self.flash_sequence)
                )

            def _on_flash_start(self, flash_index, flash_info, timestamp):
                """Called when a flash starts."""
                # Set flashing squares on board
                squares = flash_info.get('squares', [])
                self.app.board_renderer.set_flashing_squares(set(squares))

                # Send marker to live engine and simulated stream
                self.app.live_p300_engine.add_flash_event(flash_index, flash_info, timestamp)

                if self.app.simulated_stream:
                    marker = f"flash_{flash_info['type']}_{flash_info['index']}"
                    self.app.simulated_stream.send_marker(marker)

                # Add marker to visualizer
                if self.app.eeg_visualizer:
                    flash_type = flash_info.get('type', 'unknown')
                    flash_idx = flash_info.get('index', 0)
                    marker_text = f"{flash_type.upper()} {flash_idx}"
                    color = (255, 200, 100) if flash_type == 'row' else (100, 200, 255)
                    self.app.eeg_visualizer.add_marker(marker_text, color)

            def _on_flash_end(self, flash_index, flash_info, timestamp):
                """Called when a flash ends."""
                # Clear flashing
                self.app.board_renderer.clear_flash()

            def _on_sequence_complete(self):
                """Called when flash sequence completes."""
                self.state = 'processing'
                self.status_message = "Processing brain signals..."

                # Process all pending flashes
                self.app.live_p300_engine.process_pending_flashes()

                # Get results
                self.sequence_results = self.app.live_p300_engine.end_sequence()

                # Determine selected square
                self._process_selection_results()

            def _process_selection_results(self):
                """Process P300 detection results and make move."""
                if not self.sequence_results:
                    self.status_message = "Error processing results"
                    return

                row_scores = self.sequence_results['row_scores']
                col_scores = self.sequence_results['col_scores']

                # Find best row and column
                best_row = int(np.argmax(row_scores))
                best_col = int(np.argmax(col_scores))

                # Check if detection is confident enough
                row_confidence = row_scores[best_row]
                col_confidence = col_scores[best_col]

                # Calculate if scores are too uniform (no clear winner)
                row_std = np.std(row_scores)
                col_std = np.std(col_scores)

                # Detection threshold - if scores are too uniform, retry
                min_std_threshold = 0.05  # Minimum standard deviation

                detection_valid = row_std > min_std_threshold and col_std > min_std_threshold

                selected_square = chess.square(best_col, best_row)

                print(f"P300 Selection: row={best_row}, col={best_col}, square={chess.square_name(selected_square)}")
                print(f"Row scores: {row_scores} (std={row_std:.3f})")
                print(f"Col scores: {col_scores} (std={col_std:.3f})")
                print(f"Detection valid: {detection_valid}")

                if not detection_valid:
                    # No clear detection - cycle again
                    print("No clear P300 detected - cycling again...")
                    self.state = 'waiting_player'
                    self.status_message = "No clear selection detected, trying again..."
                    self.transition_time = time.time() + 1.5
                    return

                if self.selection_step == 0:
                    # Selecting origin square
                    # For testing, just pick a random legal move
                    legal_moves = self.app.game_manager.get_legal_moves()

                    if legal_moves:
                        # Pick random move
                        selected_move = random.choice(legal_moves)
                        print(f"Random move selected: {selected_move}")

                        # Make the move
                        self.app.game_manager.make_move(selected_move)
                        self.app.board_renderer.set_last_move(selected_move)

                        # Check if game is over
                        if self.app.game_manager.is_game_over():
                            self.state = 'done'
                            self.status_message = f"Game Over: {self.app.game_manager.get_result()}"
                        else:
                            # Wait a moment then start next selection
                            self.state = 'waiting_player'
                            self.status_message = "Move made! Preparing next selection..."
                            self.transition_time = time.time() + 2.0
                    else:
                        self.state = 'done'
                        self.status_message = "No legal moves!"

            def draw(self, surface):
                import pygame

                # Draw chess board
                self.app.board_renderer.draw(
                    surface,
                    self.app.game_manager.board
                )

                # Draw EEG visualizer
                if self.app.eeg_visualizer:
                    self.app.eeg_visualizer.draw(surface)

                # Draw game status
                status = self.app.game_manager.get_game_status()
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

                # Draw state-specific info
                if self.state == 'flashing':
                    # Show flash progress
                    current, total = self.app.flash_controller.get_progress()
                    progress_text = f"Flash: {current}/{total}"
                    self.app.window.draw_text(
                        progress_text,
                        (20, 60),
                        font=self.app.window.font,
                        color=(200, 200, 255)
                    )

                # Draw status message
                self.app.window.draw_text(
                    self.status_message,
                    (20, self.app.window.get_screen_size()[1] - 60),
                    font=self.app.window.large_font,
                    color=(150, 200, 255)
                )

                # Instructions
                if self.state == 'keyboard_focus':
                    instructions = "Arrow Keys: Move  |  ENTER: Confirm  |  ESC: Return to Menu"
                    # Show current focused square
                    square_text = f"Focused Square: {chess.square_name(self.focused_square).upper()}"
                    self.app.window.draw_text(
                        square_text,
                        (20, 60),
                        font=self.app.window.large_font,
                        color=(100, 255, 100)
                    )
                else:
                    instructions = "P300 Live Mode  |  ESC: Return to Menu"

                self.app.window.draw_text(
                    instructions,
                    (20, self.app.window.get_screen_size()[1] - 30),
                    font=self.app.window.font
                )

            def handle_event(self, event):
                import pygame

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Stop any ongoing sequences
                        if self.app.flash_controller.is_active():
                            self.app.flash_controller.stop()
                        self.app._show_menu()

                    # Keyboard focus mode - arrow keys to move focus
                    elif self.state == 'keyboard_focus':
                        current_file = chess.square_file(self.focused_square)
                        current_rank = chess.square_rank(self.focused_square)

                        if event.key == pygame.K_UP:
                            if current_rank < 7:
                                self.focused_square = chess.square(current_file, current_rank + 1)
                        elif event.key == pygame.K_DOWN:
                            if current_rank > 0:
                                self.focused_square = chess.square(current_file, current_rank - 1)
                        elif event.key == pygame.K_LEFT:
                            if current_file > 0:
                                self.focused_square = chess.square(current_file - 1, current_rank)
                        elif event.key == pygame.K_RIGHT:
                            if current_file < 7:
                                self.focused_square = chess.square(current_file + 1, current_rank)
                        elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                            # Confirm selection and start game
                            self.state = 'waiting_player'
                            self.status_message = "Focus set! Starting game..."
                            self.transition_time = time.time() + 1.0
                            print(f"Focus square set to: {chess.square_name(self.focused_square)}")

                        # Update highlighted square
                        self.app.board_renderer.set_highlighted_squares({self.focused_square})

            def update(self):
                # Update flash controller
                if self.app.flash_controller.is_active():
                    self.app.flash_controller.update()

                    # Process pending flashes periodically
                    if self.state == 'flashing':
                        self.app.live_p300_engine.process_pending_flashes()

                # State transitions
                if self.state == 'waiting_player':
                    if hasattr(self, 'transition_time') and time.time() >= self.transition_time:
                        # Start flash sequence
                        self._start_flash_sequence()

                elif self.state == 'error':
                    # Could add retry logic here
                    pass

        return P300GameScreen(self)

    def _create_calibration_screen(self):
        """Create calibration screen."""
        import pygame
        import random

        class CalibrationScreen:
            def __init__(self, app):
                self.app = app
                self.state = 'connecting'  # connecting, instructions, calibrating, training, done, error
                self.status_message = "Connecting to EEG stream..."

                # Calibration parameters
                self.n_targets = 4  # Number of different target squares (reduced for faster calibration)
                self.n_repetitions_per_target = self.app.config['calibration']['n_trials_per_stimulus']
                self.flash_duration = self.app.config['calibration']['flash_duration']
                self.isi = self.app.config['calibration']['isi']

                # Target squares for calibration (use chess board squares)
                self.target_squares = [
                    chess.E4, chess.D4, chess.E5, chess.D5,  # Center squares only for quick calibration
                ]
                random.shuffle(self.target_squares)
                self.current_target_idx = 0
                self.current_repetition = 0

                # Flash sequence
                self.flash_sequence = []
                self.sequence_results = None
                self.current_flash_index = 0
                self.total_flashes = 0

                # Collected data
                self.epochs = []  # List of (epoch_data, label) where label=1 for target, 0 for non-target
                self.epoch_info = []  # Metadata about each epoch

                # Prediction feedback
                self.last_prediction = None  # Last predicted square
                self.last_confidence = 0.0    # Confidence score
                self.target_confidence = 0.0  # Confidence for the actual target
                self.prediction_correct = None  # Was prediction correct?
                self.row_scores = None  # Row scores from last sequence
                self.col_scores = None  # Column scores from last sequence

                # Accuracy tracking
                self.n_correct_predictions = 0
                self.n_total_predictions = 0

                # Connect to LSL
                self._start_connection()

            def _start_connection(self):
                """Start connection to LSL stream."""
                success = self.app._connect_lsl()

                if success:
                    # Create EEG visualizer
                    if not self.app.eeg_visualizer:
                        screen_width, screen_height = self.app.window.get_screen_size()
                        board_size = self.app.config['gui']['board_size']

                        viz_x = (screen_width + board_size) // 2 + 20
                        viz_y = 50
                        viz_width = screen_width - viz_x - 20
                        viz_height = screen_height - 100

                        self.app.eeg_visualizer = EEGVisualizer(
                            n_channels=len(self.app.config['eeg']['channels']),
                            channel_names=self.app.config['eeg']['channels'],
                            sampling_rate=self.app.config['eeg']['sampling_rate'],
                            position=(viz_x, viz_y),
                            width=viz_width,
                            height=viz_height,
                            window_duration=5.0
                        )
                        print("EEG visualizer created")

                    self.state = 'instructions'
                    self.status_message = "Ready for calibration"
                    self.instruction_display_time = time.time() + 3.0
                else:
                    self.state = 'error'
                    self.status_message = "Failed to connect to EEG stream"

            def _start_calibration_trial(self):
                """Start a calibration trial."""
                if self.current_target_idx >= len(self.target_squares):
                    # All targets complete - move to training
                    self._train_classifier()
                    return

                # Get current target square
                target_square = self.target_squares[self.current_target_idx]
                target_row = chess.square_rank(target_square)
                target_col = chess.square_file(target_square)

                # Save display state for UI (what's being shown now)
                self.display_target_idx = self.current_target_idx
                self.display_repetition = self.current_repetition

                print(f"Calibration trial {self.current_target_idx + 1}/{len(self.target_squares)}, "
                      f"repetition {self.current_repetition + 1}/{self.n_repetitions_per_target}")
                print(f"Target: {chess.square_name(target_square)} (row={target_row}, col={target_col})")

                # Set target for simulated stream
                if self.app.simulated_stream:
                    self.app.simulated_stream.set_target(target_row, target_col)

                # Highlight target square
                self.app.board_renderer.set_highlighted_squares({target_square})

                # Create flash sequence
                self.flash_sequence = self.app.move_selector.create_flash_sequence()
                self.current_target_square = target_square
                self.current_target_row = target_row
                self.current_target_col = target_col

                # Reset flash tracking
                self.current_flash_index = 0
                self.total_flashes = len(self.flash_sequence)

                # Start flashing
                self.state = 'calibrating'
                self.status_message = f"Focus on {chess.square_name(target_square).upper()}..."

                self.app.flash_controller.start_sequence(
                    flash_sequence=self.flash_sequence,
                    on_flash_start=self._on_flash_start,
                    on_flash_end=self._on_flash_end,
                    on_sequence_complete=self._on_sequence_complete
                )

                # Start live P300 engine to collect data
                self.app.live_p300_engine.start_sequence(
                    sequence_id=f"calibration_{self.current_target_idx}_{self.current_repetition}",
                    n_flashes=len(self.flash_sequence)
                )

            def _on_flash_start(self, flash_index, flash_info, timestamp):
                """Called when a flash starts."""
                # Update flash progress
                self.current_flash_index = flash_index + 1  # +1 for display (1-indexed)

                squares = flash_info.get('squares', [])
                self.app.board_renderer.set_flashing_squares(set(squares))

                # Check if this flash contains the target
                is_target = self.current_target_square in squares
                label = 1 if is_target else 0

                # Record flash info for later epoch extraction
                self.epoch_info.append({
                    'timestamp': timestamp,
                    'flash_info': flash_info,
                    'label': label,
                    'target_square': self.current_target_square,
                    'extracted': False  # Track if epoch has been extracted
                })

                # Send marker to stream and engine
                self.app.live_p300_engine.add_flash_event(flash_index, flash_info, timestamp)

                if self.app.simulated_stream:
                    marker = f"flash_{flash_info['type']}_{flash_info['index']}"
                    self.app.simulated_stream.send_marker(marker)

                # Add marker to visualizer
                if self.app.eeg_visualizer:
                    flash_type = flash_info.get('type', 'unknown')
                    flash_idx = flash_info.get('index', 0)
                    marker_text = f"{flash_type.upper()} {flash_idx}"
                    color = (100, 255, 100) if is_target else (100, 100, 100)
                    self.app.eeg_visualizer.add_marker(marker_text, color)

            def _on_flash_end(self, flash_index, flash_info, timestamp):
                """Called when a flash ends."""
                self.app.board_renderer.clear_flash()

                # Extract epoch IMMEDIATELY after flash ends (before data cycles out of buffer!)
                # Wait a short time for the full epoch window to be available
                import threading
                def extract_delayed():
                    time.sleep(0.9)  # Wait for epoch window (tmax=0.8s + small buffer)

                    # Find the corresponding epoch_info entry
                    for epoch_data in self.epoch_info:
                        if epoch_data['flash_info'] == flash_info:
                            self._extract_single_epoch(epoch_data)
                            break

                # Run extraction in background to avoid blocking
                threading.Thread(target=extract_delayed, daemon=True).start()

            def _on_sequence_complete(self):
                """Called when flash sequence completes."""
                self.status_message = "Processing brain signals..."

                # Mark that we're in processing state
                self.state = 'processing'
                self.processing_start_time = time.time()

                # Mark that we need to process results (will happen in update loop)
                self.needs_processing = True

            def _analyze_prediction(self):
                """Analyze the P300 detection results and determine predicted square."""
                if not self.sequence_results:
                    return

                # Get row and column scores
                self.row_scores = self.sequence_results['row_scores']
                self.col_scores = self.sequence_results['col_scores']

                # Find best row and column
                best_row = int(np.argmax(self.row_scores))
                best_col = int(np.argmax(self.col_scores))

                # Predicted square
                self.last_prediction = chess.square(best_col, best_row)

                # Calculate confidence based on how distinct the peaks are
                # Use normalized standard deviation as a measure
                row_std = np.std(self.row_scores)
                col_std = np.std(self.col_scores)

                # Higher std means more distinct peaks (better confidence)
                # Normalize to 0-1 range (empirically, std around 0.2 is good)
                row_confidence = min(1.0, row_std / 0.2)
                col_confidence = min(1.0, col_std / 0.2)

                # Combined confidence
                self.last_confidence = (row_confidence + col_confidence) / 2.0

                # Calculate confidence for the actual target square
                target_row = chess.square_rank(self.current_target_square)
                target_col = chess.square_file(self.current_target_square)

                # Normalize scores to 0-1 range for each dimension
                row_scores_norm = (self.row_scores - self.row_scores.min()) / (self.row_scores.max() - self.row_scores.min() + 1e-10)
                col_scores_norm = (self.col_scores - self.col_scores.min()) / (self.col_scores.max() - self.col_scores.min() + 1e-10)

                # Get scores for target row and column
                target_row_score = row_scores_norm[target_row]
                target_col_score = col_scores_norm[target_col]

                # Combined target confidence
                self.target_confidence = (target_row_score + target_col_score) / 2.0

                # Check if prediction matches target
                self.prediction_correct = (self.last_prediction == self.current_target_square)

                # Update accuracy tracking
                self.n_total_predictions += 1
                if self.prediction_correct:
                    self.n_correct_predictions += 1

                # Log results
                print(f"\n=== Prediction Results ===")
                print(f"Target: {chess.square_name(self.current_target_square)} "
                      f"(row={chess.square_rank(self.current_target_square)}, "
                      f"col={chess.square_file(self.current_target_square)})")
                print(f"Predicted: {chess.square_name(self.last_prediction)} "
                      f"(row={best_row}, col={best_col})")
                print(f"Correct: {self.prediction_correct}")
                print(f"Confidence: {self.last_confidence:.2%}")
                print(f"Target Score: {self.target_confidence:.2%}")
                print(f"Row scores: {self.row_scores}")
                print(f"Col scores: {self.col_scores}")
                print(f"Target row score: {self.row_scores[target_row]:.3f} (normalized: {target_row_score:.3f})")
                print(f"Target col score: {self.col_scores[target_col]:.3f} (normalized: {target_col_score:.3f})")

                # Show online accuracy
                online_acc = (self.n_correct_predictions / self.n_total_predictions) * 100 if self.n_total_predictions > 0 else 0
                print(f"Online Accuracy: {online_acc:.1f}% ({self.n_correct_predictions}/{self.n_total_predictions})")
                print("=" * 25 + "\n")

            def _extract_single_epoch(self, epoch_meta):
                """
                Extract and process a single epoch immediately.

                Args:
                    epoch_meta: Dictionary with timestamp, label, flash_info, target_square
                """
                # Skip if already extracted
                if epoch_meta.get('extracted', False):
                    return

                epoch_tmin = self.app.config['processing']['epoch_tmin']
                epoch_tmax = self.app.config['processing']['epoch_tmax']

                timestamp = epoch_meta['timestamp']
                label = epoch_meta['label']

                # Extract epoch from buffer
                epoch_data = self.app.data_buffer.get_epoch(
                    event_timestamp=timestamp,
                    tmin=epoch_tmin,
                    tmax=epoch_tmax
                )

                if epoch_data is not None and len(epoch_data) > 0:
                    # Preprocess epoch
                    try:
                        # Validate epoch length (expected: ~250 samples for 1s epoch at 250Hz)
                        n_samples = epoch_data.shape[0]
                        expected_samples = int((epoch_tmax - epoch_tmin) * self.app.config['eeg']['sampling_rate'])

                        # Allow ±2 samples tolerance
                        if abs(n_samples - expected_samples) > 2:
                            if label == 1:
                                print(f"  ! Skipping epoch with unexpected length: {n_samples} (expected ~{expected_samples})")
                            epoch_meta['extracted'] = True  # Mark as processed but not used
                            return

                        # Filter epoch
                        if n_samples >= 30:
                            epoch_preprocessed = self.app.preprocessor.filter(epoch_data)
                        else:
                            # Skip filtering for short epochs
                            epoch_preprocessed = epoch_data
                            if label == 1:
                                print(f"  ! Target epoch too short ({n_samples} samples), skipping filter")

                        # Store raw preprocessed epoch (detector will extract features)
                        self.epochs.append((epoch_preprocessed, label))
                        epoch_meta['extracted'] = True  # Mark as extracted

                        if label == 1 and n_samples >= expected_samples - 2:
                            print(f"  ✓ Target epoch extracted: shape={epoch_data.shape}, timestamp={timestamp:.2f}")

                    except Exception as e:
                        print(f"  ✗ Error extracting epoch (n_samples={n_samples if epoch_data is not None else 'N/A'}): {e}")
                else:
                    print(f"  ✗ Failed to extract epoch at timestamp={timestamp:.2f}, label={label}")

            def _extract_epochs(self):
                """Extract epochs from buffer for the last sequence (fallback for any missed epochs)."""
                epoch_tmin = self.app.config['processing']['epoch_tmin']
                epoch_tmax = self.app.config['processing']['epoch_tmax']

                # Count already extracted vs remaining
                already_extracted = sum(1 for e in self.epoch_info if e.get('extracted', False))
                remaining = len(self.epoch_info) - already_extracted

                print(f"\n>>> Batch extraction: {already_extracted} already extracted, {remaining} remaining")

                if remaining == 0:
                    print("  All epochs already extracted immediately - skipping batch extraction")
                    return

                n_targets = 0
                n_nontargets = 0

                for idx, epoch_meta in enumerate(self.epoch_info):
                    # Skip if already extracted
                    if epoch_meta.get('extracted', False):
                        continue

                    timestamp = epoch_meta['timestamp']
                    label = epoch_meta['label']

                    # Extract epoch from buffer
                    epoch_data = self.app.data_buffer.get_epoch(
                        event_timestamp=timestamp,
                        tmin=epoch_tmin,
                        tmax=epoch_tmax
                    )

                    if idx == 0:  # Debug first epoch
                        print(f"  First epoch: timestamp={timestamp}, label={label}, data={'None' if epoch_data is None else epoch_data.shape}")
                        if epoch_data is None:
                            # Check buffer status
                            print(f"  Buffer samples written: {self.app.data_buffer.samples_written}")
                            print(f"  Buffer write index: {self.app.data_buffer.write_index}")
                            print(f"  Epoch window: [{timestamp + epoch_tmin}, {timestamp + epoch_tmax}]")
                            if self.app.data_buffer.samples_written > 0:
                                print(f"  Buffer timestamp range: [{self.app.data_buffer.timestamps[0]}, {self.app.data_buffer.timestamps[self.app.data_buffer.write_index-1]}]")

                    if epoch_data is not None and len(epoch_data) > 0:
                        # Preprocess epoch
                        try:
                            # Validate epoch length (expected: ~250 samples for 1s epoch at 250Hz)
                            n_samples = epoch_data.shape[0]
                            expected_samples = int((epoch_tmax - epoch_tmin) * self.app.config['eeg']['sampling_rate'])

                            # Allow ±2 samples tolerance
                            if abs(n_samples - expected_samples) > 2:
                                if label == 1:
                                    print(f"  ! Skipping epoch with unexpected length: {n_samples} (expected ~{expected_samples})")
                                epoch_meta['extracted'] = True  # Mark as processed but not used
                                continue

                            # Filter epoch
                            if n_samples >= 30:
                                epoch_preprocessed = self.app.preprocessor.filter(epoch_data)
                            else:
                                # Skip filtering for short epochs
                                epoch_preprocessed = epoch_data
                                print(f"  Warning: Epoch too short ({n_samples} samples), skipping filter")

                            # Store raw preprocessed epoch (detector will extract features)
                            self.epochs.append((epoch_preprocessed, label))
                            epoch_meta['extracted'] = True  # Mark as extracted

                            if label == 1:
                                n_targets += 1
                                print(f"  Target epoch collected: shape={epoch_data.shape}, mean={epoch_data.mean():.2f}, std={epoch_data.std():.2f}")
                            else:
                                n_nontargets += 1
                        except Exception as e:
                            print(f"  Warning: Skipping epoch due to error: {e}")
                            # Don't print full traceback to keep output clean
                    else:
                        if idx < 5:  # Debug first few failed extractions
                            print(f"  Epoch {idx}: Failed to extract (timestamp={timestamp}, label={label})")

                # Clear epoch info for next trial
                self.epoch_info = []

                print(f"Collected {len(self.epochs)} epochs so far (Targets: {sum(1 for _, l in self.epochs if l == 1)}, Non-targets: {sum(1 for _, l in self.epochs if l == 0)})")
                print(f"  This trial: +{n_targets} targets, +{n_nontargets} non-targets")

            def _train_classifier(self):
                """Train the P300 classifier with collected data."""
                self.state = 'training'
                self.status_message = "Training classifier..."
                print("\nTraining P300 classifier...")

                # Need at least a few epochs of each class
                n_targets = sum(1 for _, label in self.epochs if label == 1)
                n_nontargets = sum(1 for _, label in self.epochs if label == 0)

                if len(self.epochs) < 20 or n_targets < 5 or n_nontargets < 10:
                    print(f"Not enough data for training! Targets: {n_targets}, Non-targets: {n_nontargets}")
                    self.state = 'error'
                    self.status_message = "Not enough calibration data"
                    return

                # Validate epoch shapes before training
                epoch_shapes = [epoch.shape for epoch, label in self.epochs]
                unique_shapes = set(epoch_shapes)
                if len(unique_shapes) > 1:
                    print(f"Warning: Inconsistent epoch shapes detected: {unique_shapes}")
                    print(f"Filtering to most common shape...")
                    # Keep only epochs with the most common shape
                    from collections import Counter
                    most_common_shape = Counter(epoch_shapes).most_common(1)[0][0]
                    self.epochs = [(epoch, label) for epoch, label in self.epochs if epoch.shape == most_common_shape]
                    print(f"Kept {len(self.epochs)} epochs with shape {most_common_shape}")

                    # Recheck counts
                    n_targets = sum(1 for _, label in self.epochs if label == 1)
                    n_nontargets = sum(1 for _, label in self.epochs if label == 0)
                    if len(self.epochs) < 20 or n_targets < 5 or n_nontargets < 10:
                        print(f"Not enough data after filtering! Targets: {n_targets}, Non-targets: {n_nontargets}")
                        self.state = 'error'
                        self.status_message = "Not enough calibration data"
                        return

                # Prepare training data - stack epochs into 3D array
                X = np.stack([epoch for epoch, label in self.epochs])
                y = np.array([label for epoch, label in self.epochs])

                print(f"Training data: {X.shape}, Labels: {y.shape}")
                print(f"Targets: {np.sum(y == 1)}, Non-targets: {np.sum(y == 0)}")

                # Debug: Check if epochs differ between classes
                X_target = X[y == 1]
                X_nontarget = X[y == 0]
                print(f"\nEpoch analysis:")
                print(f"  Target epochs - mean: {X_target.mean():.3f}, std: {X_target.std():.3f}")
                print(f"  Non-target epochs - mean: {X_nontarget.mean():.3f}, std: {X_nontarget.std():.3f}")
                print(f"  Signal difference: {abs(X_target.mean() - X_nontarget.mean()):.3f}")
                print(f"  Epoch shape: ({X.shape[1]} samples, {X.shape[2]} channels)")

                # Train detector
                try:
                    metrics = self.app.p300_detector.train(X, y)
                    print(f"\nTraining complete! Accuracy: {metrics['cv_accuracy']:.3f}")

                    # Mark as calibrated
                    self.app.is_calibrated = True

                    # Update live engine to use trained detector
                    self.app.live_p300_engine.use_random_selection = False

                    # Auto-save calibration
                    saved_path = self.app._save_calibration()
                    if saved_path:
                        print(f"Calibration auto-saved to: {saved_path}")

                    self.state = 'done'
                    self.status_message = f"Calibration complete! Accuracy: {metrics['cv_accuracy']:.1%}"
                    self.calibration_accuracy = metrics['cv_accuracy']
                    self.done_time = time.time() + 2.0

                except Exception as e:
                    print(f"Training error: {e}")
                    import traceback
                    traceback.print_exc()
                    self.state = 'error'
                    self.status_message = f"Training failed: {str(e)}"

            def draw(self, surface):
                import pygame

                # Draw chess board
                self.app.board_renderer.draw(surface, chess.Board())

                # Draw EEG visualizer
                if self.app.eeg_visualizer:
                    self.app.eeg_visualizer.draw(surface)

                # Get board position for left-side text
                board_x, board_y = self.app.board_renderer.position
                left_x = 20
                info_x = board_x - 250

                if self.state == 'instructions':
                    # Instructions on the left
                    total_trials = len(self.target_squares) * self.n_repetitions_per_target
                    instructions = [
                        "QUICK CALIBRATION",
                        "",
                        "You will see a highlighted square",
                        "Focus your attention on it",
                        "Count how many times it flashes",
                        "",
                        f"Total trials: {total_trials}",
                        f"({len(self.target_squares)} squares x {self.n_repetitions_per_target} repetitions)",
                        "",
                        "Starting in 3 seconds..."
                    ]

                    y = 150
                    for line in instructions:
                        if line:
                            font = self.app.window.title_font if line == "QUICK CALIBRATION" else self.app.window.large_font
                            color = (255, 255, 100) if line == "QUICK CALIBRATION" else (200, 200, 200)
                            self.app.window.draw_text(line, (left_x, y), font=font, color=color)
                        y += 40

                elif self.state == 'calibrating':
                    # Progress info on the left
                    y = 100

                    # Title
                    self.app.window.draw_text(
                        "CALIBRATION",
                        (left_x, y),
                        font=self.app.window.title_font,
                        color=(255, 255, 100)
                    )
                    y += 60

                    # Current target
                    self.app.window.draw_text(
                        "Focus on square:",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(180, 180, 180)
                    )
                    y += 35
                    self.app.window.draw_text(
                        chess.square_name(self.current_target_square).upper(),
                        (left_x + 20, y),
                        font=self.app.window.title_font,
                        color=(100, 255, 100)
                    )
                    y += 60

                    # Progress - use display variables
                    total_trials = len(self.target_squares) * self.n_repetitions_per_target
                    # Count trials being displayed (current one in progress)
                    if hasattr(self, 'display_target_idx') and hasattr(self, 'display_repetition'):
                        completed_trials = (self.display_target_idx * self.n_repetitions_per_target) + self.display_repetition
                        display_square = self.display_target_idx + 1
                        display_rep = self.display_repetition + 1
                    else:
                        completed_trials = 0
                        display_square = 1
                        display_rep = 1
                    progress_pct = (completed_trials / total_trials) * 100

                    self.app.window.draw_text(
                        f"Square {display_square}/{len(self.target_squares)}",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(200, 200, 255)
                    )
                    y += 30
                    self.app.window.draw_text(
                        f"Repetition {display_rep}/{self.n_repetitions_per_target}",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(200, 200, 255)
                    )
                    y += 30
                    self.app.window.draw_text(
                        f"Total: {progress_pct:.1f}%",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(200, 200, 255)
                    )
                    y += 30

                    # Flash progress
                    self.app.window.draw_text(
                        f"Flash {self.current_flash_index}/{self.total_flashes}",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(150, 150, 200)
                    )
                    y += 50

                    # Data collected
                    self.app.window.draw_text(
                        f"Epochs: {len(self.epochs)}",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(150, 150, 150)
                    )
                    y += 30

                    # Online accuracy
                    if self.n_total_predictions > 0:
                        online_acc = (self.n_correct_predictions / self.n_total_predictions) * 100
                        acc_color = (100, 255, 100) if online_acc > 60 else (255, 200, 100) if online_acc > 40 else (255, 100, 100)
                        self.app.window.draw_text(
                            f"Online Accuracy: {online_acc:.0f}%",
                            (left_x, y),
                            font=self.app.window.font,
                            color=acc_color
                        )
                        self.app.window.draw_text(
                            f"({self.n_correct_predictions}/{self.n_total_predictions})",
                            (left_x, y + 20),
                            font=self.app.window.font,
                            color=(120, 120, 120)
                        )

                    # Show prediction feedback if available
                    if self.last_prediction is not None:
                        y += 60
                        self.app.window.draw_text(
                            "Last Prediction:",
                            (left_x, y),
                            font=self.app.window.large_font,
                            color=(180, 180, 180)
                        )
                        y += 35

                        # Show predicted square with color based on correctness
                        pred_color = (100, 255, 100) if self.prediction_correct else (255, 100, 100)
                        pred_text = f"{chess.square_name(self.last_prediction).upper()}"
                        if self.prediction_correct:
                            pred_text += " [OK]"
                        else:
                            pred_text += " [WRONG]"

                        self.app.window.draw_text(
                            pred_text,
                            (left_x + 20, y),
                            font=self.app.window.title_font,
                            color=pred_color
                        )
                        y += 45

                        # Show confidence
                        conf_color = (100, 255, 100) if self.last_confidence > 0.7 else (255, 200, 100) if self.last_confidence > 0.4 else (255, 100, 100)
                        self.app.window.draw_text(
                            f"Confidence: {self.last_confidence:.1%}",
                            (left_x, y),
                            font=self.app.window.font,
                            color=conf_color
                        )
                        y += 35

                        # Show target confidence (score for the true target)
                        target_conf_color = (100, 255, 100) if self.target_confidence > 0.7 else (255, 200, 100) if self.target_confidence > 0.4 else (255, 100, 100)
                        self.app.window.draw_text(
                            f"Target Score: {self.target_confidence:.1%}",
                            (left_x, y),
                            font=self.app.window.font,
                            color=target_conf_color
                        )

                elif self.state == 'processing':
                    # Show processing state
                    y = 100

                    # Title
                    self.app.window.draw_text(
                        "CALIBRATION",
                        (left_x, y),
                        font=self.app.window.title_font,
                        color=(255, 255, 100)
                    )
                    y += 60

                    # Processing message
                    self.app.window.draw_text(
                        "Analyzing P300 response...",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(200, 200, 255)
                    )
                    y += 40

                    # Show current progress
                    total_trials = len(self.target_squares) * self.n_repetitions_per_target
                    completed_trials = (self.current_target_idx * self.n_repetitions_per_target) + self.current_repetition
                    progress_pct = (completed_trials / total_trials) * 100

                    self.app.window.draw_text(
                        f"Progress: {progress_pct:.1f}%",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(150, 150, 150)
                    )

                elif self.state == 'waiting':
                    # Show progress while waiting
                    y = 100

                    # Title
                    self.app.window.draw_text(
                        "CALIBRATION",
                        (left_x, y),
                        font=self.app.window.title_font,
                        color=(255, 255, 100)
                    )
                    y += 60

                    # Progress
                    total_trials = len(self.target_squares) * self.n_repetitions_per_target
                    completed_trials = (self.current_target_idx * self.n_repetitions_per_target) + self.current_repetition
                    progress_pct = (completed_trials / total_trials) * 100

                    self.app.window.draw_text(
                        f"Progress: {progress_pct:.1f}%",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(200, 200, 255)
                    )
                    y += 30
                    self.app.window.draw_text(
                        f"Epochs: {len(self.epochs)}",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(150, 150, 150)
                    )
                    y += 50

                    # Show last prediction
                    if self.last_prediction is not None:
                        self.app.window.draw_text(
                            "Last Prediction:",
                            (left_x, y),
                            font=self.app.window.large_font,
                            color=(180, 180, 180)
                        )
                        y += 35

                        # Show predicted square with color based on correctness
                        pred_color = (100, 255, 100) if self.prediction_correct else (255, 100, 100)
                        pred_text = f"{chess.square_name(self.last_prediction).upper()}"
                        if self.prediction_correct:
                            pred_text += " [OK]"
                        else:
                            # Use saved previous target
                            if hasattr(self, 'previous_target_square'):
                                pred_text += f" [WRONG] (was {chess.square_name(self.previous_target_square).upper()})"
                            else:
                                pred_text += " [WRONG]"

                        self.app.window.draw_text(
                            pred_text,
                            (left_x + 20, y),
                            font=self.app.window.large_font,
                            color=pred_color
                        )
                        y += 40

                        # Show confidence
                        conf_color = (100, 255, 100) if self.last_confidence > 0.7 else (255, 200, 100) if self.last_confidence > 0.4 else (255, 100, 100)
                        self.app.window.draw_text(
                            f"Confidence: {self.last_confidence:.1%}",
                            (left_x, y),
                            font=self.app.window.font,
                            color=conf_color
                        )
                        y += 30

                        # Show target confidence
                        target_conf_color = (100, 255, 100) if self.target_confidence > 0.7 else (255, 200, 100) if self.target_confidence > 0.4 else (255, 100, 100)
                        self.app.window.draw_text(
                            f"Target Score: {self.target_confidence:.1%}",
                            (left_x, y),
                            font=self.app.window.font,
                            color=target_conf_color
                        )
                        y += 50

                    self.app.window.draw_text(
                        "Get ready for next trial...",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(200, 200, 200)
                    )

                elif self.state == 'training':
                    # Training message on left
                    y = 200
                    self.app.window.draw_text(
                        "Training Classifier...",
                        (left_x, y),
                        font=self.app.window.title_font,
                        color=(255, 255, 100)
                    )
                    y += 60
                    self.app.window.draw_text(
                        f"{len(self.epochs)} epochs collected",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(200, 200, 200)
                    )
                    y += 40

                    # Breakdown
                    X = np.array([epoch for epoch, label in self.epochs])
                    y_labels = np.array([label for epoch, label in self.epochs])
                    self.app.window.draw_text(
                        f"Targets: {np.sum(y_labels == 1)}",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(100, 255, 100)
                    )
                    y += 30
                    self.app.window.draw_text(
                        f"Non-targets: {np.sum(y_labels == 0)}",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(150, 150, 150)
                    )

                elif self.state == 'done':
                    # Success message on left
                    y = 150
                    self.app.window.draw_text(
                        "Calibration Complete!",
                        (left_x, y),
                        font=self.app.window.title_font,
                        color=(100, 255, 100)
                    )
                    y += 60
                    self.app.window.draw_text(
                        self.status_message,
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(200, 200, 200)
                    )
                    y += 80

                    # Mode selection
                    self.app.window.draw_text(
                        "Select test mode:",
                        (left_x, y),
                        font=self.app.window.large_font,
                        color=(255, 255, 100)
                    )
                    y += 50

                    self.app.window.draw_text(
                        "K - Keyboard Test (manual control)",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(200, 200, 200)
                    )
                    y += 35

                    self.app.window.draw_text(
                        "P - P300 Mode (BCI control)",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(200, 200, 200)
                    )
                    y += 35

                    self.app.window.draw_text(
                        "ESC - Return to menu",
                        (left_x, y),
                        font=self.app.window.font,
                        color=(150, 150, 150)
                    )

                # Status message at bottom
                self.app.window.draw_text(
                    self.status_message,
                    (20, self.app.window.get_screen_size()[1] - 60),
                    font=self.app.window.large_font,
                    color=(150, 200, 255)
                )

                # Instructions
                instructions = "Calibration Mode  |  ESC: Cancel"
                self.app.window.draw_text(
                    instructions,
                    (20, self.app.window.get_screen_size()[1] - 30),
                    font=self.app.window.font
                )

            def handle_event(self, event):
                import pygame

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Cancel calibration or return to menu
                        if self.app.flash_controller.is_active():
                            self.app.flash_controller.stop()
                        self.app._show_menu()

                    elif event.key == pygame.K_k and self.state == 'done':
                        # Start keyboard test mode
                        print("Starting keyboard test mode...")
                        self.app.game_mode = 'demo'  # Use demo mode for keyboard control
                        self.app._start_game()

                    elif event.key == pygame.K_p and self.state == 'done':
                        # Start P300 mode
                        print("Starting P300 BCI mode...")
                        self.app.game_mode = 'live'
                        self.app._start_game()

            def update(self):
                # Update flash controller
                if self.app.flash_controller.is_active():
                    self.app.flash_controller.update()

                # Process pending flashes during calibration
                if self.state == 'calibrating':
                    self.app.live_p300_engine.process_pending_flashes()

                # State transitions
                if self.state == 'instructions':
                    if time.time() >= self.instruction_display_time:
                        self._start_calibration_trial()

                elif self.state == 'processing':
                    elapsed = time.time() - self.processing_start_time

                    # Extract epochs IMMEDIATELY (don't wait - data will cycle out of buffer!)
                    if hasattr(self, 'needs_processing') and self.needs_processing:
                        # Process pending flashes (non-blocking)
                        self.app.live_p300_engine.process_pending_flashes()

                        # Check if processing is complete
                        if self.app.live_p300_engine.is_processing_complete():
                            # Get P300 detection results (non-blocking)
                            self.sequence_results = self.app.live_p300_engine.end_sequence_nonblocking()

                            if self.sequence_results:
                                # Analyze prediction
                                self._analyze_prediction()

                                self.needs_processing = False
                                self.needs_extraction = True
                        elif elapsed >= 2.0:
                            # Timeout - something went wrong
                            print("Warning: P300 processing timeout!")
                            self.sequence_results = self.app.live_p300_engine.end_sequence_nonblocking()
                            if self.sequence_results:
                                self._analyze_prediction()
                            self.needs_processing = False
                            self.needs_extraction = True

                    # Then extract epochs (after processing completes)
                    if hasattr(self, 'needs_extraction') and self.needs_extraction and not self.needs_processing:
                        # Extract epochs for this trial
                        self._extract_epochs()
                        self.needs_extraction = False

                        # Save current target square for display
                        self.previous_target_square = self.current_target_square
                        current_square_name = chess.square_name(self.current_target_square).upper()

                        # Move to next trial
                        self.current_repetition += 1
                        if self.current_repetition >= self.n_repetitions_per_target:
                            self.current_repetition = 0
                            self.current_target_idx += 1
                            self.status_message = f"Target {current_square_name} complete! Moving to next..."
                        else:
                            self.status_message = f"Repetition {self.current_repetition}/{self.n_repetitions_per_target} for {current_square_name}"

                        # Update display progress (for next trial)
                        self.display_target_idx = self.current_target_idx
                        self.display_repetition = self.current_repetition

                        # Clear highlights
                        self.app.board_renderer.clear_highlights()

                        # Wait before next trial
                        self.state = 'waiting'
                        self.wait_until = time.time() + 0.5  # Shorter wait since we already waited in processing

                elif self.state == 'waiting':
                    if hasattr(self, 'wait_until') and time.time() >= self.wait_until:
                        self._start_calibration_trial()

                elif self.state == 'done':
                    # Wait for user to select mode (K for keyboard, P for P300, ESC for menu)
                    pass

        return CalibrationScreen(self)

    def _connect_lsl(self) -> bool:
        """
        Connect to LSL stream (or start simulated stream).

        Returns:
            True if successful, False otherwise
        """
        use_simulated = self.config['eeg'].get('use_simulated', False)

        if use_simulated:
            # Start simulated EEG stream
            print("Starting simulated EEG stream...")
            try:
                stream_name = self.config['eeg'].get('simulated_stream_name', 'SimulatedEEG')
                self.simulated_stream = SimulatedEEGStream(
                    stream_name=stream_name,
                    n_channels=len(self.config['eeg']['channels']),
                    sampling_rate=self.config['eeg']['sampling_rate'],
                    channel_names=self.config['eeg']['channels']
                )
                self.simulated_stream.start()
                time.sleep(1.0)  # Give stream time to start

                # Connect LSL client to simulated stream
                self.lsl_client.stream_name = stream_name
                if self.lsl_client.connect(timeout=5.0):
                    self.lsl_client.start_acquisition(self._on_eeg_sample)
                    print("Connected to simulated stream successfully")
                    return True
                else:
                    print("Failed to connect to simulated stream")
                    return False

            except Exception as e:
                print(f"Error starting simulated stream: {e}")
                return False
        else:
            # Connect to real LSL stream
            print("Connecting to real LSL stream...")
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

        # Update visualizer if active
        if self.eeg_visualizer:
            self.eeg_visualizer.add_sample(sample, timestamp)

    def _save_calibration(self, stream_id: str = None):
        """
        Save calibration data to file.

        Args:
            stream_id: EEG stream identifier (defaults to simulated/real stream name)
        """
        if not self.is_calibrated or not self.p300_detector.is_trained:
            print("Cannot save: No calibration data available")
            return None

        # Create calibrations directory
        calib_dir = Path("calibrations")
        calib_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp and stream ID
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if stream_id is None:
            stream_id = self.config['eeg'].get('simulated_stream_name' if self.config['eeg']['use_simulated'] else 'stream_name', 'UnknownStream')

        filename = f"calibration_{timestamp}_{stream_id}.pkl"
        filepath = calib_dir / filename

        # Prepare calibration data
        calib_data = {
            'timestamp': datetime.now().isoformat(),
            'stream_id': stream_id,
            'detector': self.p300_detector,  # Save entire detector object
            'preprocessor_params': {
                'bandpass_low': self.config['processing']['bandpass_low'],
                'bandpass_high': self.config['processing']['bandpass_high'],
                'notch_freq': self.config['processing']['notch_freq'],
            },
            'config': self.config,
        }

        # Save to file
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(calib_data, f)
            print(f"Calibration saved: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return None

    def _load_calibration(self, filepath: str):
        """
        Load calibration data from file.

        Args:
            filepath: Path to calibration file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                calib_data = pickle.load(f)

            print(f"\nLoading calibration from: {filepath}")
            print(f"  Timestamp: {calib_data['timestamp']}")
            print(f"  Stream ID: {calib_data['stream_id']}")

            # Restore detector
            self.p300_detector = calib_data['detector']

            # Mark as calibrated
            self.is_calibrated = True

            # Update live engine to use the loaded detector
            self.live_p300_engine.p300_detector = self.p300_detector
            self.live_p300_engine.use_random_selection = False

            print("Calibration loaded successfully!")
            print(f"  Detector method: {self.p300_detector.method}")
            print(f"  Training accuracy: {self.p300_detector.training_metrics.get('cv_accuracy', 0):.3f}")
            return True

        except Exception as e:
            print(f"Error loading calibration: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _cleanup(self):
        """Cleanup resources."""
        print("\nShutting down...")

        # Auto-save calibration if available
        if self.is_calibrated:
            print("Auto-saving calibration...")
            self._save_calibration()

        # Stop EEG acquisition
        if self.lsl_client:
            self.lsl_client.stop_acquisition()

        # Stop simulated stream
        if self.simulated_stream:
            self.simulated_stream.stop()

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
