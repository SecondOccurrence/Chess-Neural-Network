import torch
from torch import nn

from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
  def __init__(self, input_size, output_size, lr):
    super().__init__()

    self.input_size = input_size
    self.output_size = output_size
    self.lr = lr

    self.net: nn.Module = self.create_network(self.output_size)
    self.initialise_weights()

  @abstractmethod
  def create_network(self, output_size) -> nn.Module:
    pass

  def initialise_weights(self):
    # iterate through every layer in the model
    for layer in self.children():
      if isinstance(layer, nn.Linear):
        # initialise weights and bias to zero
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        torch.nn.init.zeros_(layer.bias)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), self.lr)

  def load_state(self, weight_path):
    state = None
    try:
      state = torch.load(weight_path, weights_only=True)
    except FileNotFoundError:
      print(f"Weight path: {weight_path} does not exist. Model state will not be loaded")

    if state is not None:
      self.load_state_dict(state)
      print("Successfully loaded the model state")
    else:
      print("Failed to load the model state. Program will close.")
      exit(1)

  def forward(self, X):
    return self.net(X)

  def loss(self, y_hat, y):
    # TODO: explain why using cross entropy loss
    fn = nn.CrossEntropyLoss()
    return fn(y_hat, y)
