from torch import nn

from .base_model import BaseModel
from .residual import Residual

class ResNet18(BaseModel):
  def create_network(self, output_size) -> nn.Module:
    self.architecture = [(2, 64), (2, 128), (2, 256), (2, 512)]

    self.net = nn.Sequential(
      nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
      nn.LazyBatchNorm2d(),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    for i, b in enumerate(self.architecture):
      self.net.add_module(f"block_{i+2}", self.block(*b, first_block=(i==0)))

    self.net.add_module("last", nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
      nn.LazyLinear(output_size)
    ))

    return self.net

  def block_1(self):
    return nn.Sequential(
      nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
      nn.LazyBatchNorm2d(),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

  def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
      if i == 0 and not first_block:
        blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
      else:
        blk.append(Residual(num_channels))

    return nn.Sequential(*blk)

