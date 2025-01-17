import torch
from config import config
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from callbacks import LogMetricsCallback, WeightsSaver

config_train_params = config['train_params']

#empty, only function names init and train

class Train:
    def __init__(self, model_object, train_dataset, dev_dataset):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model_object

        models_dir = './models'
        self.model_name = config['model_name']
        self.summary_dir = os.path.join(models_dir, self.model_name, "summaries")
        self.checkpoint_dir = os.path.join(models_dir, self.model_name, "checkpoints")

        os.makedirs(os.path.join(self.summary_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.summary_dir, "dev"), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_writer = SummaryWriter(os.path.join(self.summary_dir, "train"))
        self.dev_writer = SummaryWriter(os.path.join(self.summary_dir, "dev"))

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.model.to(device)
        optimizer = Adam(self.model.parameters(), lr=0.0001)
        criterion = CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pth")
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1

        tboard = LogMetricsCallback(self.train_writer, self.dev_writer, summary_step=10)
        checkpoint_callback = WeightsSaver(self.checkpoint_dir, latest_checkpoint_step=10)

        num_epochs = config_train_params['epochs']
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            self.model.train()
            train_loss, train_accuracy = self.run_epoch(
                self.train_dataset, device, optimizer, criterion, tboard, epoch, "train"
            )

            self.model.eval()
            with torch.no_grad():
                dev_loss, dev_accuracy = self.run_epoch(
                    self.dev_dataset, device, None, criterion, tboard, epoch, "dev"
                )

            checkpoint_callback.save(epoch, self.model, optimizer)

            print(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
                f"Dev Loss = {dev_loss:.4f}, Dev Acc = {dev_accuracy:.4f}"
            )
            scheduler.step(train_loss)

    def run_epoch(self, dataloader, device, optimizer, criterion, tboard, epoch, phase):
        total_loss, correct, total = 0.0, 0, 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            if phase == "train" and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                accuracy = correct / total if total > 0 else 0
                print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        tboard.on_batch_end(loss.item(), accuracy, phase)

        accuracy = correct / total if total > 0 else 0
        return total_loss, accuracy
