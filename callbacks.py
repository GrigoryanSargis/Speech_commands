import torch
from config import config
import os

class LogMetricsCallback:
    """
    this class is for tensorboard summaries
    implement __init__() and on_train_batch_end() functions
    you should be able to save summaries with config['train_params']['summary_step'] frequency
    tensorboard should show loss and accuracy for train and validation separately
    those in their respective folders defined in train
    """
    def __init__(self, train_writer, dev_writer, summary_step):
        """
        Args:
            train_writer (SummaryWriter): TensorBoard writer for training logs.
            dev_writer (SummaryWriter): TensorBoard writer for validation/dev logs.
            summary_step (int): Frequency of logging metrics.
        """
        self.train_writer = train_writer
        self.dev_writer = dev_writer
        self.summary_step = summary_step

        self.train_step = 0
        self.dev_step = 0

    def on_batch_end(self, loss, accuracy, phase='train'):
        if phase == 'train':
            self.train_writer.add_scalar('Loss', loss, self.train_step)
            self.train_writer.add_scalar('Accuracy', accuracy, self.train_step)
            self.train_step += 1

        elif phase == 'dev':
            self.dev_writer.add_scalar("Loss", loss, self.dev_step)
            self.dev_writer.add_scalar('Accuracy', accuracy, self.dev_step)
            self.dev_step += 1

    def close(self):
        """
        Closes the TensorBoard writers.
        """
        self.train_writer.close()
        self.dev_writer.close()



class WeightsSaver:
    """
    this class is for checkpoints
    implement __init__ and on_train_batch_end functions and any other auxilary functions you may need
    it should be able to save at  config['train_params']['latest_checkpoint_step']
    it should save 'max_to_keep' number of checkpoints EX. if max_to_keep = 5, you should keep only 5 newest checkpoints
    save in the folder defined in train 
    """
    def __init__(self, save_dir, latest_checkpoint_step, max_to_keep=5):
        """
        Args:
            save_dir (str): Directory to save checkpoints.
            latest_checkpoint_step (int): Frequency of saving checkpoints.
            max_to_keep (int): Maximum number of checkpoints to retain.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.latest_checkpoint_step = latest_checkpoint_step
        self.max_to_keep = max_to_keep
        self.checkpoints = []

    def save(self, epoch, model, optimizer):
        """
        Saves a checkpoint for the given epoch.
        Args:
            epoch (int): Current epoch.
            model (nn.Module): Model to save.
            optimizer (torch.optim.Optimizer): Optimizer to save.
        """
        if epoch % self.latest_checkpoint_step == 0:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, checkpoint_path)

            self.checkpoints.append(checkpoint_path)

            if len(self.checkpoints) > self.max_to_keep:
                oldest_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"Removed old checkpoint: {oldest_checkpoint}")

            print(f"Saved checkpoint: {checkpoint_path}")

    def load_latest_checkpoint(self, model, optimizer):
        """
        Loads the latest checkpoint if available.
        Args:
            model (nn.Module): Model to load.
            optimizer (torch.optim.Optimizer): Optimizer to load.
        Returns:
            int: The epoch number of the loaded checkpoint, or 0 if no checkpoint found.
        """
        if self.checkpoints:
            # Sort checkpoints by epoch to ensure correct order
            self.checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = self.checkpoints[-1]
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"Loaded checkpoint: {latest_checkpoint}")
            return checkpoint['epoch']
        
        print("No checkpoints available to load.")
        return 0
