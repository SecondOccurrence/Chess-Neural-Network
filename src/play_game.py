import chess

from .config import Config
from .data import ChessUtils
from .model.resnet18 import ResNet18

def print_board(board):
  """
  Displays the board in a human readable format

  Args:
    board (chess.Board): the board state to display

  """

  white = "WHITE"
  black = "BLACK"

  print(f"{black: ^23}")
  print("  +-----------------+")

  for rank in range(8, 0, -1):
    row = f"{rank} |"
    for file in range(8):
      piece = board.piece_at(file + (rank - 1) * 8)
      row += f" {piece.symbol() if piece else '.'}"
    row += f" |"
    print(row)

  print("  +-----------------+")
  print("    a b c d e f g h")
  print(f"{white: ^23}")

def nn_move(model, board):
  print("Execute Black move")

  input = ChessUtils.board_to_matrix(board)
  output = model(ChessUtils.to_model_compatible(input))
  current_score = ChessUtils.extract_true_score(output)

  possible_moves = list(board.legal_moves)
  possible_scores = []

  temp_board = board.copy()
  for move in possible_moves:
    # Apply the possible move to a temporary board state
    temp_board.push(move)
    temp_input = ChessUtils.board_to_matrix(temp_board)

    # Retrieve the models score interpretation of the board state
    temp_output = model(ChessUtils.to_model_compatible(temp_input))
    temp_score = ChessUtils.extract_true_score(temp_output)

    possible_scores.append(temp_score)

    # Restore the board state before applying move
    temp_board = board.copy()

  best_move_index = ChessUtils.find_best_score(current_score, possible_scores)   
  best_move = possible_moves[best_move_index]
  board.push(best_move)

def main():
  conf = Config(data_path="./data/chess_dataset.npz")
  model = ResNet18(input_size=conf.input_size, output_size=conf.output_size, lr=conf.nn.learning_rate, lr_gamma=conf.nn.lr_gamma).eval()
  model.load_state(conf.nn.weight_path)

  board = chess.Board()
  print_board(board)

  while not board.is_game_over():
    player = "White" if board.turn == chess.WHITE else "Black"
    if player == "Black":
      nn_move(model, board)
      print_board(board)
    else: 
      print(f"Your move")

      move_input = input("Enter move in UCI format (e2e4): ")

      if move_input == "exit":
        break

      try:
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
          board.push(move)
          print_board(board)
      except Exception as _:
        print("Please enter a valid move")

  if board.is_checkmate():
    print(f"{player} won")
  elif board.is_stalemate():
    print("Stalemate")
  else:
    print("Game over")

if __name__ == "__main__":
  main()
