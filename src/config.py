import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from data.chess_dataset import ChessDataset
from training.transforms import *

from dataclasses import dataclass

@dataclass
class ModelParameters:
  weight_path: str
  retrain_model: bool
  batch_size: int
  learning_rate: float
  lr_gamma: float
  num_epochs: int


class Config:
  def __init__(self, data_path):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.nn = ModelParameters(
      weight_path = "../saves/chess_nn_weights.pth",
      retrain_model=True,
      batch_size=1024,
      learning_rate=0.001,
      lr_gamma=0.1,
      num_epochs=100
    )

    self.data_transform = transforms.Compose([
      transforms.ToTensor(),
      RandomTransform(HorizontalFlip(), probability=0.5),
      RandomTransform(VerticalFlip(), probability=0.5),
    ])

    self.dataset = ChessDataset(data_path=data_path, data_transform=self.data_transform)
    self.input_size = 14*8*8
    self.output_size = 1

    print(f"Input size: {self.input_size}")
    print(f"Output size: {self.output_size}")

    self.train_data, self.val_data, self.test_data = self.__get_dataset()
    self.train_loader = DataLoader(self.train_data, batch_size=self.nn.batch_size, shuffle=True)
    self.val_loader = DataLoader(self.val_data, batch_size=self.nn.batch_size, shuffle=False)
    self.test_loader = DataLoader(self.test_data, batch_size=self.nn.batch_size, shuffle=True)

  def __get_dataset(self):
    """
    Splits the dataset into
    70% training
    20% validation
    10% testing

    """
    data_size = len(self.dataset)
    train_size = int(0.7 * data_size)
    val_size = int(0.2 * data_size)
    test_size = data_size - train_size - val_size

    return random_split(self.dataset, [train_size, val_size, test_size])
