"""
Main application window using Pygame.
Coordinates calibration, game play, and visualization.
"""

import pygame
import sys
from typing import Optional, Callable


class MainWindow:
    def __init__(self, config: dict):
        """
        Initialize main window.

        Args:
            config: Configuration dictionary from settings.yaml
        """
        self.config = config

        # Initialize Pygame
        pygame.init()

        # Create window
        if config['gui']['fullscreen']:
            self.screen = pygame.display.set_mode(
                (config['gui']['screen_width'], config['gui']['screen_height']),
                pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode(
                (config['gui']['screen_width'], config['gui']['screen_height'])
            )

        pygame.display.set_caption("P300 Chess BCI")

        # Colors
        self.bg_color = tuple(config['gui']['background_color'])
        self.flash_color = tuple(config['gui']['flash_color'])
        self.text_color = (220, 220, 220)

        # Fonts
        self.font = pygame.font.Font(None, config['gui']['font_size'])
        self.large_font = pygame.font.Font(None, int(config['gui']['font_size'] * 1.5))
        self.title_font = pygame.font.Font(None, int(config['gui']['font_size'] * 2))

        # Clock for timing
        self.clock = pygame.time.Clock()

        # Current screen/mode
        self.current_screen = None
        self.running = True

        # Event callbacks
        self.on_quit = None

    def set_screen(self, screen):
        """
        Change current screen.

        Args:
            screen: Screen object with update(), draw(), and handle_event() methods
        """
        self.current_screen = screen

    def run(self):
        """Main application loop."""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    if self.on_quit:
                        self.on_quit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        if self.on_quit:
                            self.on_quit()

                # Pass events to current screen
                if self.current_screen and hasattr(self.current_screen, 'handle_event'):
                    self.current_screen.handle_event(event)

            # Update current screen
            if self.current_screen and hasattr(self.current_screen, 'update'):
                self.current_screen.update()

            # Draw
            self.screen.fill(self.bg_color)

            if self.current_screen and hasattr(self.current_screen, 'draw'):
                self.current_screen.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()

    def stop(self):
        """Stop the main loop."""
        self.running = False

    def draw_text(self, text: str, position: tuple, font: pygame.font.Font = None,
                  color: tuple = None, center: bool = False):
        """
        Draw text on screen.

        Args:
            text: Text to draw
            position: (x, y) position
            font: Font to use (default: self.font)
            color: Text color (default: self.text_color)
            center: If True, center text at position
        """
        if font is None:
            font = self.font
        if color is None:
            color = self.text_color

        text_surface = font.render(text, True, color)

        if center:
            text_rect = text_surface.get_rect(center=position)
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, position)

    def draw_text_lines(self, lines: list, start_position: tuple,
                       line_spacing: int = 30, font: pygame.font.Font = None,
                       color: tuple = None):
        """
        Draw multiple lines of text.

        Args:
            lines: List of text lines
            start_position: (x, y) starting position
            line_spacing: Spacing between lines in pixels
            font: Font to use
            color: Text color
        """
        x, y = start_position
        for line in lines:
            self.draw_text(line, (x, y), font=font, color=color)
            y += line_spacing

    def draw_button(self, text: str, rect: pygame.Rect, hover: bool = False) -> pygame.Rect:
        """
        Draw a button.

        Args:
            text: Button text
            rect: Button rectangle
            hover: If True, draw hover state

        Returns:
            Button rectangle
        """
        # Button colors
        button_color = (60, 60, 60) if not hover else (80, 80, 80)
        border_color = (100, 100, 100) if not hover else (120, 120, 120)

        # Draw button background
        pygame.draw.rect(self.screen, button_color, rect)
        pygame.draw.rect(self.screen, border_color, rect, 2)

        # Draw text
        text_surface = self.font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

        return rect

    def show_message(self, title: str, message: str, duration: float = 3.0):
        """
        Show a temporary message overlay.

        Args:
            title: Message title
            message: Message text
            duration: Duration to show in seconds
        """
        import time
        start_time = time.time()

        while time.time() - start_time < duration:
            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            # Draw semi-transparent overlay
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            # Draw message box
            box_width = 600
            box_height = 200
            box_rect = pygame.Rect(
                (self.screen.get_width() - box_width) // 2,
                (self.screen.get_height() - box_height) // 2,
                box_width,
                box_height
            )

            pygame.draw.rect(self.screen, (40, 40, 40), box_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), box_rect, 2)

            # Draw title
            title_surface = self.large_font.render(title, True, (255, 255, 255))
            title_rect = title_surface.get_rect(
                centerx=box_rect.centerx,
                y=box_rect.y + 30
            )
            self.screen.blit(title_surface, title_rect)

            # Draw message
            msg_surface = self.font.render(message, True, (200, 200, 200))
            msg_rect = msg_surface.get_rect(center=box_rect.center)
            self.screen.blit(msg_surface, msg_rect)

            pygame.display.flip()
            self.clock.tick(60)

    def get_screen_center(self) -> tuple:
        """Get center coordinates of screen."""
        return (
            self.screen.get_width() // 2,
            self.screen.get_height() // 2
        )

    def get_screen_size(self) -> tuple:
        """Get screen dimensions."""
        return (
            self.screen.get_width(),
            self.screen.get_height()
        )
