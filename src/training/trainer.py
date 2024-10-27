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
    self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.model.lr_gamma, patience=10)

  def train(self, train_loader, val_loader, epochs, device):
    self.model.train()

    best_val_loss = float("inf")
    val_patience = 0
    val_threshold = 0.01
    for epoch in range(epochs):
      print(f"Epoch {epoch + 1} / {epochs}")
      self.fit_epoch(train_loader, epoch, device)
      val_loss = self.validate(val_loader, epoch, device)

      self.scheduler.step(val_loss)

      if val_loss > best_val_loss + val_threshold:
        val_patience += 1
        if val_patience >= 8:
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
      loss = self.model.loss(outputs, labels)
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

    # Store the number of correct guesses
    correct = 0
    total = 0

    with torch.no_grad():
      for inputs, labels in val_loader:
        # TODO: label must be converted to max index
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = self.model(inputs)
        loss = self.model.loss(outputs, labels)
        total_loss += loss.item()

        # Extract the class that the model has chosen as the highest probability
        #   We don't need the first of the return value, so discarding
        _, predicted_labels = torch.max(outputs.data, 1)
        _, actual_labels = torch.max(labels.data, 1)
        # Add the batch size of the data to the total
        total += labels.size(0)

        correct_labels = (predicted_labels == actual_labels).sum().item()
        correct += correct_labels

    # Calculate statistics
    average_loss = total_loss / len(val_loader)
    model_accuracy = correct / total
    as_percentage = model_accuracy * 100

    # Print statistics and write for tensorboard
    print(f", validation_loss: {average_loss:.3f}, validation_accuracy: {as_percentage:.2f}%\n")
    self.writer.add_scalar("Loss/Valid", average_loss, epoch)
    self.writer.add_scalar("Loss/Accuracy", model_accuracy, epoch)

    return average_loss

  def test(self, test_loader, batch_size, device):
    self.model.eval()

    test_loss = 0
    correct_predictions = 0
    with torch.no_grad():
      for (inputs, target) in test_loader:
        inputs = inputs.to(device)
        target = target.to(device)

        outputs = self.model(inputs)

        # Find the class with the highest predicted probability
        _, predicted = torch.max(outputs.data, 1)
        # Compare predicted class to target class (0 or 1) in the batch
        correct_predictions += (predicted == target).sum().item()

    num_batches = len(test_loader)
    test_loss /= num_batches
    accuracy = correct_predictions / (num_batches * batch_size)

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
