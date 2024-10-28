from torch.utils.data import Dataset

import numpy as np

class ChessDataset(Dataset):
  def __init__(self, data_path, data_transform=None):
    """
    Arguments:
      data_path (str): Path to the .npz dataset
      data_transform (callable, optional): Optional transform to be applied on a sample

    """

    self.dataset = []
    self.__load_dataset(filename=data_path)
    self.data_transform = data_transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    board, score = self.dataset[index]

    # transforms.ToTensor() seems to expect the input to be of shape (H, W, C)
    #   this input is in (C, H, W), so we need to convert it before transforming
    board = np.transpose(board, (1, 2, 0))
    if self.data_transform is not None:
      board = self.data_transform(board)

    board = board.float()

    return board, score

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
    board_score = loaded_data['targets']

    if(len(chess_board) != len(board_score)):
      print("Number of chess boards are not equal to the number of moves. Data loaded is invalid. Data will not be loaded.")
      return

    # Reset dataset related variables in case any data has already been loaded
    self.dataset = []
    self.all_possible_moves = set()

    for board_state, target in zip(chess_board, board_score):
      self.dataset.append((board_state, target))

    print(f"Successfully loaded {len(self.dataset)} samples.")
