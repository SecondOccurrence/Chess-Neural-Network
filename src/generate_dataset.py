from .data import ChessDataGenerator

data_generator = ChessDataGenerator(num_samples=100000, stockfish_path="../../stockfish", save_path="../../data/chess_dataset.npz")
data_generator.generate()

