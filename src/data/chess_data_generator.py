import sys
import chess
import chess.engine
import random
import numpy as np

from .utils import ChessUtils

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

    engine1, engine2 = self.__init_players()

    board = chess.Board()

    self.update_progress(len(self.dataset))

    save_counter = 0
    while len(self.dataset) < self.num_samples:
      if board.is_game_over():
        # Create stockfish player of different skill levels for a more diverse dataset
        engine1.quit()
        engine2.quit()
        engine1, engine2 = self.__init_players()
        # Reset board to start
        board = chess.Board()

      if board.turn == chess.WHITE:
        current_player = engine1
      else:
        current_player = engine2

      sf_next_move = self.__get_move(current_player, board)

      if sf_next_move is None:
        # Reset board if no move for some reason
        engine1.quit()
        engine2.quit()
        engine1, engine2 = self.__init_players()
        board = chess.Board()
        continue
      
      board.push(sf_next_move)

      # Evaluate the state of the board
      sf_result = current_player.analyse(board, chess.engine.Limit(depth=3), info=chess.engine.INFO_SCORE)
      score = self.__retrieve_score(sf_result)

      if score is None:
        print("Error on retrieval of dataset label (board score)")
        exit(0)

      board_as_matrix = ChessUtils.board_to_matrix(board)

      # Store (np.ndarray, float32) in temporary dataset variable
      self.dataset.append((board_as_matrix, score))

      self.update_progress(len(self.dataset))

      save_counter += 1
      # Save at intervals in case an unexpected error
      if save_counter % 499 == 0:
        # Save the dataset
        self.__save_dataset(filename=self.save_path)
        save_counter = 0

    engine1.quit()
    engine2.quit()
    self.__save_dataset(filename=self.save_path)

  def __init_players(self):
    e1_skill = random.randint(3, 18)
    # Making sure there is a level playing field
    #  So both engines have a realistic chance of winning
    e2_skill = random.randint(3, e1_skill + 2)

    engine1 = self.__configure_stockfish(skill_level=e1_skill)
    engine2 = self.__configure_stockfish(skill_level=e2_skill)

    return engine1, engine2

  def __get_move(self, player, board):
    # Random % chance between 0% and 40% of the move being random
    random_move_chance = 0.15

    if random.random() < random_move_chance:
      move = random.choice(list(board.legal_moves))
    else:
      move = player.play(board, chess.engine.Limit(time=0.1)).move

    return move

  def __configure_stockfish(self, skill_level=20):
    try:
      engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
      engine.configure({"Skill Level": skill_level})
    except chess.engine.EngineTerminatedError:
      sys.stderr.write(f"Engine terminated unexpectedly. Check the stockfish path: \"{self.stockfish_path}\"\n")

    return engine

  def __retrieve_score(self, board_state):
    """
    Retrieves a score that represents the board state.

    To be used as the label in a dataset sample

    """

    score = board_state.get("score")

    if score:
      if score.is_mate():
        # This means it returns a mate score. To convert mate score into centipawn score, you pass in this optional parameter
        score_value = np.float32(score.relative.score(mate_score=9999))
      else:
        score_value = np.float32(score.relative.score())
    else:
      return None

    return score_value
        
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
