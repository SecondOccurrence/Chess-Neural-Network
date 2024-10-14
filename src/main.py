import numpy as np

from data.chess_data_generator import ChessDataGenerator

data_generator = ChessDataGenerator()

print("Do you fish to load the dataset?")

load_data = input()
if load_data == 'y':
  data_generator.load_dataset(filename="../data/chess_dataset.npz")
else:
  data_generator.generate();
  data_generator.save_dataset(filename="../data/chess_dataset.npz")

#print(f"Move index: {test_index}")
#test_move = list(data_generator.all_possible_moves)[test_index]
#print(f"Actual move: {test_move}")
