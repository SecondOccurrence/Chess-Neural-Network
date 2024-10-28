import chess
import numpy as np

class ChessUtils:
  @staticmethod
  def __square_to_coord(square: int) -> tuple[int, int]:
    """
    Converts a number representation of a chess board square (0-63) to a coordinate in a 8x8 grid

    Args:
      square (int): number to convert to coordinate

    Returns:
      tuple[int, int]: in range 0..7,0..7

    """

    row = square // 8
    column = square % 8

    return (row, column)
  
  @staticmethod
  def board_to_matrix(board) -> np.ndarray:
    """
    Breaks down a chess board state into a 3d matrix representation

    The 3d matrix is of size 14*8*8. This represents 14 matrices of a chess board (8*8=64 squares).
    The first 6 matrices show the white sides pieces (as 1) in order PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING.
    For example, the 3rd matrix shows the white sides BISHOPS on the board
    The next 6 matrices show the black side pieces (as 1) in the same order as above
    The next 2 matrices are dedicated to white side and black side possible moves, where a possible move is noted as 1, and others 0

    Args:
      board (chess.Board): The board state to convert into the 3d matrix representation

    Returns:
      np.ndarray: A chess board in 3d matrix representation of dimension 14*8*8

    """

    matrix = np.zeros((14, 8, 8), dtype=np.int8)

    for piece in chess.PIECE_TYPES:
      # Add to white matrices (0-5)
      for square in board.pieces(piece, chess.WHITE):
        coords = ChessUtils.__square_to_coord(square)
        matrix[piece - 1][coords[0]][coords[1]] = 1
      # Add to black matrices (6-11)
      for square in board.pieces(piece, chess.BLACK):
        coords = ChessUtils.__square_to_coord(square)
        matrix[piece + 5][coords[0]][coords[1]] = 1
        
    # Save board turn to restore later
    saved_turn = board.turn

    # Add possible white moves (12)
    board.turn = chess.WHITE
    for move in board.legal_moves:
      coords = ChessUtils.__square_to_coord(move.to_square)
      matrix[12][coords[0]][coords[1]] = 1

    # Add possible black moves (13)
    board.turn = chess.BLACK
    for move in board.legal_moves:
      coords = ChessUtils.__square_to_coord(move.to_square)
      matrix[13][coords[0]][coords[1]] = 1

    # Restore previous turn
    board.turn = saved_turn

    return matrix
