from torch.utils.data import Dataset

import numpy as np
import chess

class ChessDataset(Dataset):
  def __init__(self, data_path, data_transform=None):
    """
    Arguments:
      data_path (str): Path to the .npz dataset
      data_transform (callable, optional): Optional transform to be applied on a sample

    """
    self.dataset = []
    self.all_possible_moves = set()
    self.__load_dataset(filename=data_path)
    self.data_transform = data_transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    board, move_vector = self.dataset[index]

    # transforms.ToTensor() seems to expect the input to be of shape (H, W, C)
    #   this input is in (C, H, W), so we need to convert it before transforming
    board = np.transpose(board, (1, 2, 0))
    if self.data_transform is not None:
      board = self.data_transform(board)

    board = board.float()
    return board, move_vector

  def __load_dataset(self, filename):
    """
    Loads a dataset (saved as .npz) into the class

    This will contain the actual dataset (containing the chess board states as 3d array, and best move index)
    along with all the possible moves in the dataset

    Note that self.dataset & self.all_possible_moves will be reset

    Args:
      filename (str): The path at which the file to load is located

    """

    print(f"Loading dataset: \"{filename}\"")
    loaded_data = np.load(filename, allow_pickle=True)

    chess_board = loaded_data['boards']
    best_move_indices = loaded_data['targets']
    possible_moves = loaded_data['possible_moves']

    if(len(chess_board) != len(best_move_indices)):
      print("Number of chess boards are not equal to the number of moves. Data loaded is invalid. Data will not be loaded.")
      return

    # Reset dataset related variables in case any data has already been loaded
    self.dataset = []
    self.all_possible_moves = set()

    for move in possible_moves:
      self.all_possible_moves.add(move)

    for board_state, target in zip(chess_board, best_move_indices):
      self.dataset.append((board_state, target))

    print(f"Successfully loaded {len(self.dataset)} samples.")
    print(f"Successfully loaded {len(self.all_possible_moves)} possible moves.")
