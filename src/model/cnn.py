from torch import nn

from .base_model import BaseModel

class ChessNeuralNetwork(BaseModel):
  def create_network(self) -> nn.Module:
    # TODO: Research on best layers combination / network style
    # TODO: Research best activation function
    #   TODO: one at the end of the output layer?
    return nn.Sequential()
