import torch
import random

class RandomTransform:
  """
  Applies a custom transformation,
  but with a percent chance of applying the transformation

  """
  def __init__(self, transform, probability=0.5):
    self.transform = transform
    self.probability = probability

  def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
    if random.random() < self.probability:
      return self.transform(tensor)
    else:
      return tensor


class HorizontalFlip:
  """
  A custom transformation on a 3D tensor.

  Considering my input tensor is of shape [14, 8, 8], I believe that a standard horizontal flip
  provided by pytorch would flip the board incorrectly, causing the board to change, and thus becoming invalid.

  Here I simply flip the 3rd dimension only.

  """
  def __call__(self, tensor):
    return tensor.flip(2)

class VerticalFlip:
  """
  A custom transformation on a 3D tensor.

  Flipping the board verticalls (only the third dimension)

  """
  def __call__(self, tensor):
    return tensor.flip(1)
