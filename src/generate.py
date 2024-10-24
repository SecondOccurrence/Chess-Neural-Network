from data.chess_data_generator import ChessDataGenerator

data_generator = ChessDataGenerator(num_samples=250000, stockfish_path="../stockfish", save_path="../data/chess_dataset.npz")
data_generator.generate()

