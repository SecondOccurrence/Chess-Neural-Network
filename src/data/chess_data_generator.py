import sys
import chess
import chess.engine
import numpy as np
from typing import Optional

class ChessDataGenerator:
  """
  A class to manage dataset generation and saving

  Attributes:
    num_samples (int): Number of chess board states to be generated for the dataset
    stockfish_path (str): relative path to the stockfish executable
    temp_dataset (numpy.ndarray, chess.Move): Stores the dataset before converting the chess move into a target/label for the NN
    dataset (numpy.ndarray, numpy.ndarray): The dataset.
      [0]: (14*8*8) representation of a board state. Explained in board_to_matrix(..)
      [1]: The score of the board state. < 0 for black side favor, > 0 white side favor

  """

  def __init__(self, num_samples: int, stockfish_path: str, save_path: str):
    self.num_samples = num_samples
    self.stockfish_path = stockfish_path
    self.save_path = save_path

    self.dataset = []

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

    self.update_progress(len(self.dataset))

    save_counter = 0
    while len(self.dataset) < self.num_samples:
      if board.is_game_over():
        # Reset board on game end
        board = chess.Board()

      sf_next_move = engine.play(board, chess.engine.Limit(time=0.1))
      if sf_next_move is None:
        # Reset board if no move for some reason
        board = chess.Board()
        continue
      else:
        best_move: chess.Move = sf_next_move.move
        board.push(best_move)

      # Evaluate the state of the board
      sf_result = engine.analyse(board, chess.engine.Limit(depth=2))
      
      score = self.__retrieve_score(sf_result)
      if score is None:
        print("Error on retrieval of dataset label (board score)")
        exit(0)

      board_as_matrix = self.board_to_matrix(board)

      # Store (np.ndarray, float32) in temporary dataset variable
      self.dataset.append((board_as_matrix, score))

      self.update_progress(len(self.dataset))

      save_counter += 1
      # Save at intervals in case an unexpected error
      if save_counter % 499 == 0:
        # Save the dataset
        self.__save_dataset(filename=self.save_path)
        save_counter = 0

    engine.quit()
    self.__save_dataset(filename=self.save_path)


  def __retrieve_score(self, board_state):
    """
    Retrieves a score that represents the board state.

    To be used as the label in a dataset sample

    """

    score = board_state.get("score")

    if score:
      if score.is_mate():
        # Considering white side scores are positive values and black negative
        score_value = np.float32(9999) if score.is_mate() > 0 else np.float32(-9999)
      else:
        score_value = np.float32(score.relative.score())
    else:
      return None

    return score_value
        
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
      
  def update_progress(self, completed_samples):
    bar_length = 30

    progress = completed_samples / self.num_samples
    bar_progress = int(bar_length * progress)

    bar = '#' * bar_progress + '-' * (bar_length - bar_progress)

    progress_display = f"{completed_samples}/{self.num_samples}"
    print(f"\r[{bar}] {progress:.2%}, {progress_display:15}", end="")

  def __save_dataset(self, filename):
    """
    Saves the dataset into a numpy .npz file
    Converts the dataset target (chess.Move) into an index pointing to that move
    in all moves previously generated by stockfish.
    This will help in specifying the Neural Network's output layer

    Dataset consists of:
    1. the board state as 3d matrix defined in board_to_matrix(..)
    2. the best move for that board as an index in a list of all possible moves the dataset has generated

    Args:
      filename (str): The save path of the file

    """

    boards = []
    targets = []

    for board_matrix, best_move in self.dataset:
      boards.append(board_matrix)
      targets.append(best_move)

    boards = np.array(boards)
    targets = np.array(targets)

    np.savez_compressed(filename, boards=boards, targets=targets)
    print(f"Dataset saved to \"{filename}\"")
