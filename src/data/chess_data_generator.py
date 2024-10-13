class ChessDataGenerator:
  def __init__(self, num_samples=2048, stockfish_path="../stockfish"):
    self.num_samples = num_samples
    self.stockfish_path = stockfish_path
