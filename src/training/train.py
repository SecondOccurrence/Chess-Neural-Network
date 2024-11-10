from ..config import Config
from ..model import ResNet18
from .trainer import Trainer

conf = Config(data_path="./data/chess_dataset.npz")

def train_model(model, train_loader, val_loader, save_path, retrain, epochs, device, model_name):
  print(f"Do you wish to train the {model_name} model? [Y,y/N,n]")

  to_train = input()
  if to_train.lower() != 'y':
    return

  if retrain is False:
    model.load_state(save_path)

  model.to(device)
  trainer = Trainer(model, save_path)
  trainer.train(train_loader, val_loader, epochs, device)

chess_nn_model = ResNet18(conf.input_size, conf.output_size, conf.nn.learning_rate, conf.nn.lr_gamma)
train_model(
  chess_nn_model, conf.train_loader, conf.val_loader,
  conf.nn.weight_path, conf.nn.retrain_model,
  conf.nn.num_epochs, conf.device, "Chess Neural Network"
)
