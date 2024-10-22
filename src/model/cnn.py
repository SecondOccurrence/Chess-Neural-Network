from torch import nn

from .base_model import BaseModel

class ChessNeuralNetwork(BaseModel):
  def create_network(self, output_size) -> nn.Module:
    return nn.Sequential(
      nn.Conv2d(14, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

      nn.Flatten(),
      nn.Linear(64*2*2, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, output_size)
    )
