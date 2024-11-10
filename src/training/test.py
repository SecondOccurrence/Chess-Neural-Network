from ..config import Config
from ..model.resnet18 import ResNet18
from .trainer import Trainer

def main():

  conf = Config(data_path="./data/chess_dataset.npz")

  model = ResNet18(input_size=conf.input_size, output_size=conf.output_size, lr=conf.nn.learning_rate, lr_gamma=conf.nn.lr_gamma)
  model.load_state(conf.nn.weight_path);

  model.eval()
  model.to(conf.device)

  print("Model testing..")
  trainer = Trainer(model, conf.nn.weight_path)
  trainer.test(conf.test_loader, conf.device)

if __name__ == "__main__":
  main()
