from torch.utils.data import Dataset
from typing import Optional,List, Tuple

import numpy as np

class ChessDataset(Dataset):
  def __init__(self, data_path, data_transform=None):
    """
    Arguments:
      data_path (str): Path to the .npz dataset
      data_transform (callable, optional): Optional transform to be applied on a sample

    """

    self.min_score = -9999
    self.max_score = 9999

    self.loaded_data = self.__load_dataset(filename=data_path)
    if self.loaded_data is None:
      return

    self.boards = [board for board, _ in self.loaded_data]
    self.scores = np.array([score for _, score in self.loaded_data])

    self.normalised_scores = self.__normalise_scores(self.scores)

    self.data_transform = data_transform

  def __normalise_scores(self, scores):
    """
    Returns the normalised labels of a dataset in range [-1, 1]

    """

    return 2 * (scores - self.min_score) / (self.max_score - self.min_score) - 1

  def get_score(self, index):
    return self.scores[index]

  def __len__(self):
    if self.loaded_data is None:
      length = 0
    else:
      length = len(self.loaded_data)

    return length

  def __getitem__(self, index):
    board = self.boards[index]
    score = self.normalised_scores[index]

    # transforms.ToTensor() seems to expect the input to be of shape (H, W, C)
    #   this input is in (C, H, W), so we need to convert it before transforming
    board = np.transpose(board, (1, 2, 0))
    if self.data_transform is not None:
      board = self.data_transform(board)

    board = board.float()

    return board, score

  def __load_dataset(self, filename) -> Optional[List[Tuple[np.ndarray, float]]]:
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
      return None

    # Reset dataset related variables in case any data has already been loaded
    dataset = []
    self.all_possible_moves = set()

    for board_state, target in zip(chess_board, board_score):
      dataset.append((board_state, target))

    print(f"Successfully loaded {len(dataset)} samples.")
    return dataset
