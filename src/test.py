import chess
import chess.svg
import cairosvg

from config import Config
from model.resnet18 import ResNet18

"""
  compare scores for the board
"""
def test_model(model, dataloader, device):
  model.eval()
  model.to(device)

  for i, (board, _) in enumerate(dataloader):
    board.to(device)
    output = model(board)


conf = Config(data_path="../data/chess_dataset_250k.npz")

model = ResNet18(input_size=conf.input_size, output_size=conf.output_size, lr=conf.nn.learning_rate, lr_gamma=conf.nn.lr_gamma)
model.load_state(conf.nn.weight_path);

test_model(model, conf.test_loader, conf.device)

"""
board = chess.Board();

svg_board = chess.svg.board(
  board,
  fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
  arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
  squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
  size=350,
)

with open("chess_board.svg", "w") as svg_file:
  svg_file.write(svg_board)

cairosvg.svg2png(url="chess_board.svg", write_to="chess_board.png")
"""
