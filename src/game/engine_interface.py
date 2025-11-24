"""
Interface to chess engines (Stockfish initially, extensible to others).
"""

import chess
import chess.engine
from typing import Optional, Dict
import os


class ChessEngineInterface:
    def __init__(self, engine_path: str = None,
                 skill_level: int = 5, depth: int = 10):
        """
        Initialize chess engine.

        Args:
            engine_path: Path to engine executable (None for auto-detect)
            skill_level: Skill level (0-20 for Stockfish)
            depth: Search depth
        """
        # Try to find Stockfish if path not provided
        if engine_path is None:
            engine_path = self._find_stockfish()

        self.engine_path = engine_path
        self.skill_level = skill_level
        self.depth = depth
        self.engine = None
        self.is_running = False

    def _find_stockfish(self) -> str:
        """
        Try to find Stockfish executable in common locations.

        Returns:
            Path to Stockfish or raises FileNotFoundError
        """
        common_paths = [
            '/usr/games/stockfish',
            '/usr/local/bin/stockfish',
            '/usr/bin/stockfish',
            'stockfish',  # In PATH
            '/opt/homebrew/bin/stockfish',  # MacOS Homebrew
        ]

        for path in common_paths:
            if os.path.exists(path) or os.system(f'which {path} > /dev/null 2>&1') == 0:
                return path

        raise FileNotFoundError(
            "Stockfish not found. Please install Stockfish or provide engine_path. "
            "Install with: sudo apt-get install stockfish (Ubuntu/Debian) or "
            "brew install stockfish (MacOS)"
        )

    def start(self):
        """Start the chess engine."""
        if self.is_running:
            print("Engine already running")
            return

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

            # Configure Stockfish-specific options
            try:
                self.engine.configure({"Skill Level": self.skill_level})
                print(f"Stockfish started with skill level {self.skill_level}")
            except chess.engine.EngineError:
                # Not Stockfish or doesn't support Skill Level
                print(f"Engine started (skill level not supported)")

            self.is_running = True

        except Exception as e:
            raise RuntimeError(f"Failed to start engine: {e}")

    def stop(self):
        """Stop the chess engine."""
        if self.engine and self.is_running:
            try:
                self.engine.quit()
            except:
                pass
            self.is_running = False
            print("Engine stopped")

    def get_best_move(self, board: chess.Board,
                     time_limit: Optional[float] = None) -> Optional[chess.Move]:
        """
        Get best move for current position.

        Args:
            board: Current board state
            time_limit: Optional time limit in seconds (overrides depth)

        Returns:
            Best move according to engine
        """
        if not self.is_running:
            raise RuntimeError("Engine not started")

        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=self.depth)

        result = self.engine.play(board, limit)
        return result.move

    def analyze_position(self, board: chess.Board,
                        time_limit: Optional[float] = None) -> Dict:
        """
        Analyze position and return evaluation.

        Args:
            board: Current board state
            time_limit: Optional time limit in seconds

        Returns:
            Dictionary with score and principal variation
        """
        if not self.is_running:
            raise RuntimeError("Engine not started")

        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=self.depth)

        info = self.engine.analyse(board, limit)

        # Extract score
        score = info.get('score')
        if score:
            # Convert to centipawns from white's perspective
            score_cp = score.white().score(mate_score=10000)
        else:
            score_cp = None

        return {
            'score_cp': score_cp,
            'score': info.get('score'),
            'pv': info.get('pv', []),
            'depth': info.get('depth', 0),
            'nodes': info.get('nodes', 0),
            'time': info.get('time', 0)
        }

    def set_skill_level(self, level: int):
        """
        Set engine skill level (Stockfish only).

        Args:
            level: Skill level (0-20)
        """
        if not self.is_running:
            raise RuntimeError("Engine not started")

        self.skill_level = level
        try:
            self.engine.configure({"Skill Level": level})
            print(f"Skill level set to {level}")
        except chess.engine.EngineError:
            print("Skill level not supported by this engine")

    def set_depth(self, depth: int):
        """
        Set search depth.

        Args:
            depth: Search depth in plies
        """
        self.depth = depth
        print(f"Search depth set to {depth}")

    def get_engine_info(self) -> Dict:
        """
        Get engine information.

        Returns:
            Dictionary with engine info
        """
        if not self.is_running:
            return {
                'name': 'Not started',
                'running': False
            }

        return {
            'name': self.engine.id.get('name', 'Unknown'),
            'author': self.engine.id.get('author', 'Unknown'),
            'running': self.is_running,
            'skill_level': self.skill_level,
            'depth': self.depth
        }

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
