from data.chess_data_generator import ChessDataGenerator

data_generator = ChessDataGenerator()

data_generator.generate();
data_generator.save_dataset(filename="../data/chess_dataset.npz")
