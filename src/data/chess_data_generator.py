import sys
import chess
import chess.engine
import numpy as np

class ChessDataGenerator:
  def __init__(self, num_samples=2048, stockfish_path="../stockfish"):
    self.num_samples = num_samples
    self.stockfish_path = stockfish_path
    self.dataset = []

  def generate(self):
    try:
      engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
    except chess.engine.EngineTerminatedError:
      sys.stderr.write(f"Engine terminated unexpectedly. Check the stockfish path: \"{self.stockfish_path}\"\n")
      return

    board = chess.Board()

    while len(self.dataset) < self.num_samples and not board.is_game_over():
      sf_result = engine.play(board, chess.engine.Limit(time=0.5))
      if sf_result is None:
        print("No valid move found. Ending generation.")
        break

      best_move: chess.Move = sf_result.move
      board_as_matrix = self.board_to_matrix(board)
      print(board_as_matrix)

      self.dataset.append((board_as_matrix, best_move))

      board.push(best_move)

    engine.quit()

  def board_to_matrix(self, board) -> np.ndarray:
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
      for square in board.pieces(piece, chess.WHITE):
        coords = self.__square_to_coord(square)
        matrix[piece - 1][coords[0]][coords[1]] = 1
      for square in board.pieces(piece, chess.BLACK):
        coords = self.__square_to_coord(square)
        matrix[piece + 5][coords[0]][coords[1]] = 1

    return matrix

  def __square_to_coord(self, square: int) -> tuple[int, int]:
    if square < 0 or square > 63:
      raise ValueError("Number must be between 0 and 63.")

    row = square // 8
    column = square % 8

    return (row, column)




    # 14 8 8


#board = chess.Board()
#example_move = self.engine.play(board, chess.engine.Limit(time=1.0))
#print(example_move)

#  for each sample:
#    get a random board state
#    use stockfish to generate the next best possible move
#    convert board state to 3d matrix representation
#    add (board_as_matrix, best_move) to dataset[]

