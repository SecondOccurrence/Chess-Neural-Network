import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter

class Trainer:
  def __init__(self, model, save_path):
    # Create object for tensorboard logging throughout the training phase
    self.writer = SummaryWriter()
    self.model = model
    self.save_path = save_path
    self.optimizer = model.configure_optimizers()
    self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.model.lr_gamma, patience=5)

  def train(self, train_loader, val_loader, epochs, device):
    self.model.train()

    best_val_loss = float("inf")
    val_patience = 0
    val_threshold = 0.06
    for epoch in range(epochs):
      print(f"Epoch {epoch + 1} / {epochs}")
      self.fit_epoch(train_loader, epoch, device)
      val_loss = self.validate(val_loader, epoch, device)

      self.scheduler.step(val_loss)

      if val_loss > best_val_loss + val_threshold:
        val_patience += 1
        if val_patience >= 5:
          print("Stopping training early. Validation loss is increasing too much")
          break
      elif val_loss < best_val_loss:
        best_val_loss = val_loss
        print("New best loss achieved. Saving model state.")
        torch.save(self.model.state_dict(), self.save_path)

    print("Training process has finished")
    self.writer.flush()
    self.writer.close()

  def fit_epoch(self, train_loader, epoch, device):
    torch.cuda.empty_cache()

    current_loss = 0.0
    total_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
      # Send the data to the right device
      inputs = inputs.to(device)
      labels = labels.to(device)

      self.optimizer.zero_grad()

      # Perform a forward pass through the network
      outputs = self.model(inputs)

      # Calculate loss between outputs and target,
      #   Perform backpropagation
      #   and update parameters
      loss = self.model.loss(outputs.squeeze(), labels)
      loss.backward()

      self.optimizer.step()

      current_loss += loss.item()
      total_loss += current_loss

      total_batches = len(train_loader)
      # Prints statistics on training progress
      self.update_progress(i, total_batches, current_loss)
      current_loss = 0.0

    average_loss = total_loss / total_batches
    self.writer.add_scalar("Loss/Train", average_loss, epoch)

  def update_progress(self, current_batch, total_batches, loss):
    bar_length = 30
    progress = (current_batch + 1) / total_batches
    bar_progress = int(bar_length * progress)

    bar = '#' * bar_progress + '-' * (bar_length - bar_progress)
    print(f"\r[{bar}] {progress:.2%} | train_loss: {loss:.4f}", end="")

  def validate(self, val_loader, epoch, device):
    # Set the model to evaluation mode for consistency
    self.model.eval()

    # Store the total loss of the set
    total_loss = 0
    total_samples = 0
    total_error_margins = 0.0
    accurate_predictions = 0

    threshold = 1

    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = self.model(inputs).squeeze()

        loss = self.model.loss(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        error_margins = torch.abs(outputs - labels)
        total_error_margins += error_margins.sum().item()

        accurate_predictions += (error_margins <= threshold).sum().item()

    # Calculate statistics
    average_loss = total_loss / total_samples
    mean_abs_error = total_error_margins / total_samples
    accuracy = accurate_predictions / total_samples * 100

    # Print statistics and write for tensorboard
    print(f", validation_loss: {average_loss:.3f}, MAE: {mean_abs_error:.3f}, validation_accuracy: {accuracy:.2f}%\n")
    self.writer.add_scalar("Loss/Valid", average_loss, epoch)
    self.writer.add_scalar("Loss/Accuracy", accuracy, epoch)

    return average_loss

  def test(self, test_loader, device):
    self.model.eval()

    # Store the total loss of the set
    total_loss = 0
    total_samples = 0
    total_error_margins = 0.0
    accurate_predictions = 0

    threshold = 1

    with torch.no_grad():
      for (inputs, target) in test_loader:
        inputs, labels = inputs.to(device), target.to(device)

        outputs = self.model(inputs).squeeze()

        loss = self.model.loss(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        error_margins = torch.abs(outputs - labels)
        total_error_margins += error_margins.sum().item()

        accurate_predictions += (error_margins <= threshold).sum().item()

    # Calculate statistics
    average_loss = total_loss / total_samples
    mean_abs_error = total_error_margins / total_samples
    accuracy = accurate_predictions / total_samples * 100

    print(f"Test Loss: {average_loss:.3f}, MAE: {mean_abs_error:.3f}, Validation Accuracy: {accuracy:.2f}%\n")
    print(f"Out of {total_samples} total samples..")
    print(f"Model correctly determined {accurate_predictions} samples. ({accurate_predictions}/{total_samples})")
