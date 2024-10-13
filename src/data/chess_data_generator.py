import sys
import chess
import chess.engine
import numpy as np

class ChessDataGenerator:
  def __init__(self, num_samples=2048, stockfish_path="../stockfish"):
    self.num_samples = num_samples
    self.stockfish_path = stockfish_path
    self.dataset = []
    self.all_possible_moves = set()

  def generate(self):
    """
    Generates a dataset containing board states and their best moves

    Stores the dataset in self.dataset
    If the self.stockfish_path is invalid the function will return early and print a warning to stderr

    """
    try:
      engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
    except chess.engine.EngineTerminatedError:
      sys.stderr.write(f"Engine terminated unexpectedly. Check the stockfish path: \"{self.stockfish_path}\"\n")
      return

    board = chess.Board()

    self.update_progress()
    while len(self.dataset) < self.num_samples:
      if board.is_game_over():
        # Reset board on game end
        board = chess.Board()

      sf_result = engine.play(board, chess.engine.Limit(time=0.1))
      if sf_result is None:
        # Reset board if no move for some reason
        board = chess.Board()
        continue

      best_move: chess.Move = sf_result.move
      board_as_matrix = self.board_to_matrix(board)

      self.dataset.append((board_as_matrix, best_move))
      self.all_possible_moves.add(best_move)

      board.push(best_move)

      self.update_progress()

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
      # Add to white matrices (0-5)
      for square in board.pieces(piece, chess.WHITE):
        coords = self.__square_to_coord(square)
        matrix[piece - 1][coords[0]][coords[1]] = 1
      # Add to black matrices (6-11)
      for square in board.pieces(piece, chess.BLACK):
        coords = self.__square_to_coord(square)
        matrix[piece + 5][coords[0]][coords[1]] = 1
        
      # Save board turn to restore later
    saved_turn = board.turn

    # Add possible white moves (12)
    board.turn = chess.WHITE
    for move in board.legal_moves:
      coords = self.__square_to_coord(move.to_square)
      matrix[12][coords[0]][coords[1]] = 1

    # Add possible black moves (13)
    board.turn = chess.BLACK
    for move in board.legal_moves:
      coords = self.__square_to_coord(move.to_square)
      matrix[13][coords[0]][coords[1]] = 1

    # Restore previous turn
    board.turn = saved_turn

    return matrix

  def __square_to_coord(self, square: int) -> tuple[int, int]:
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

  def update_progress(self):
    bar_length = 30
    completed_samples = len(self.dataset)

    progress = completed_samples / self.num_samples
    bar_progress = int(bar_length * progress)

    bar = '#' * bar_progress + '-' * (bar_length - bar_progress)
    print(f"\r[{bar}] {progress:.2%}, {completed_samples}/{self.num_samples}", end="")

  def save_dataset(self, filename="dataset.npz"):
    """
    ** apply softmax during inference or loss calculation after output layer
    ** use cross entropy
    Saves the dataset into a numpy .npz file

    Dataset consists of:
    1. the board state as 3d matrix defined in board_to_matrix(..)
    2. the best move for that board as an index in a list of all possible moves the dataset has generated

    """

    num_moves = len(self.all_possible_moves)
    moves_as_list = list(self.all_possible_moves)

    move_indices = {}
    for index, move in enumerate(moves_as_list):
      move_indices[move] = index

    boards = []
    targets = []

    for board_matrix, best_move in self.dataset:
      boards.append(board_matrix)

      current_target = np.zeros(num_moves)
      index = move_indices[best_move]
      current_target[index] = 1

      targets.append(current_target)

    boards = np.array(boards)
    targets = np.array(targets)

    np.savez_compressed(filename, boards=boards, targets=targets)
    print(f"Dataset successfully saved into \"{filename}\"")
