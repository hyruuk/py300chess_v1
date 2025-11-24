# Keyboard Demo Mode

Test the P300 Chess BCI system without EEG hardware using keyboard controls!

## Quick Start

1. **Ensure demo mode is enabled** in `config/settings.yaml`:
   ```yaml
   demo:
     enabled: true
     keyboard_control: true
     auto_play_engine: true
   ```

2. **Run the application**:
   ```bash
   python src/main.py
   ```

3. **Press G** to start the game

## Keyboard Controls

### Moving the Cursor
- **Arrow Keys (↑ ↓ ← →)**: Move cursor around the board
- The cursor is shown as a dark green highlighted square

### Selecting Moves (Two-Step Process)

#### Step 1: Select Origin Square
1. Use arrow keys to move cursor to the piece you want to move
2. Press **SPACE** to select that piece
3. All legal destination squares will be highlighted in yellow-green

#### Step 2: Select Destination
1. Use **UP/DOWN arrows** to cycle through legal moves
2. The cursor automatically moves to show the destination square
3. Press **SPACE** to confirm the move

### Other Controls
- **ESC**:
  - If in step 2: Cancel selection and return to step 1
  - If in step 1: Return to main menu
- **M**: Return to menu (during game)

## Visual Indicators

| Color | Meaning |
|-------|---------|
| Dark Green | Current cursor position |
| Yellow-Green | Legal destination squares |
| Yellow | Last move made |
| Light Green | Previously highlighted squares |

## Example Gameplay

### Opening Move (e2-e4)

1. **Start**: Cursor at e2 (default)
   - Board shows white pawns and pieces

2. **Press SPACE**: Select e2 pawn
   - Legal moves highlighted: e3, e4
   - Cursor jumps to e4 (first option)

3. **Press SPACE again**: Confirm e4
   - Pawn moves from e2 to e4
   - Engine automatically responds (if enabled)

### Selecting Between Multiple Options

If a piece has multiple legal moves:

1. **Select the piece** (SPACE)
2. **Use UP/DOWN** to cycle through destinations
3. **Watch the cursor** move to each option
4. **Press SPACE** when you see the move you want

## Features

### Automatic Engine Opponent
When `auto_play_engine: true`, the chess engine automatically plays black's moves after you play white's move.

**To disable**: Set `auto_play_engine: false` in config

### Legal Move Validation
The system only allows legal moves:
- Can't select empty squares (in step 1)
- Can't select squares with opponent pieces (in step 1)
- Only shows legal destinations for selected piece
- Handles special moves (castling, en passant, promotion)

### Pawn Promotion
When promoting a pawn:
- Automatically selects Queen promotion
- (Future: Add UI to choose piece type)

## Comparison with P300 Mode

| Feature | Keyboard Demo | P300 BCI |
|---------|--------------|----------|
| **Input** | Arrow keys + Spacebar | EEG signals |
| **Selection** | 2-step (piece → move) | Row/column flashing |
| **Speed** | Instant | ~20-30 seconds per move |
| **Setup** | No hardware needed | Requires EEG system |
| **Accuracy** | 100% | ~70-90% (depends on calibration) |

## Tips

1. **Learn the Interface**: Practice a few moves to get comfortable with the two-step selection

2. **Watch the Status Text**: The top of the screen shows whether you're in step 1 (select origin) or step 2 (select destination)

3. **Use ESC to Undo**: If you select the wrong piece, press ESC to go back

4. **Observe Engine Moves**: Watch how the engine responds to learn chess strategy

5. **Full Games**: Play complete games to test all chess rules (check, checkmate, stalemate)

## Testing Scenarios

### Basic Movement
- Move pawns forward (e2-e4, d2-d4)
- Develop knights (Ng1-f3)
- Castle (after moving pieces)

### Special Situations
- **Check**: When king is in check, only legal moves to escape are available
- **Promotion**: Move pawn to 8th rank (automatic Queen)
- **Castling**: King + Rook special move
- **En Passant**: Special pawn capture

### Game Endings
- **Checkmate**: Opponent's king has no legal moves
- **Stalemate**: No legal moves but not in check
- **Draw**: (implement in future - repetition, 50-move rule)

## Troubleshooting

### Cursor doesn't move
- Make sure demo mode is enabled in config
- Check that keyboard has focus (click the window)

### Can't select a square
- Verify there's a piece on that square
- Ensure it's your turn (white moves first)

### No legal moves shown
- The selected piece may have no legal moves
- Press ESC and try a different piece

### Engine doesn't respond
- Check that Stockfish is installed
- Look for error messages in terminal
- Set `auto_play_engine: false` to play both sides

## Configuration Options

Edit `config/settings.yaml`:

```yaml
demo:
  enabled: true              # Enable/disable demo mode
  keyboard_control: true     # Use keyboard instead of P300
  auto_play_engine: true     # Engine plays automatically
  show_legal_moves: true     # Highlight legal destinations

chess:
  engine_skill_level: 5      # 0 (easy) to 20 (very hard)
  engine_depth: 10           # How deep engine searches

gui:
  fullscreen: false          # Windowed mode for testing
  board_size: 640           # Board size in pixels
```

## Next Steps

After testing with keyboard:
1. Understand the two-step selection process
2. See how it maps to row/column flashing in P300 mode
3. Install EEG hardware
4. Run calibration
5. Play chess with brain signals!

## Support

Having issues? Check:
- Terminal output for error messages
- Config file syntax (YAML formatting)
- Stockfish installation (`which stockfish`)

For P300 mode, see README.md
