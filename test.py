import os
import torch
from config import config
 #empty
#only comments
class Test:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def test(self):
        """
        Perform inference on the test dataset.
        Ensure that checkpoints directory is not empty,
        restore from the latest checkpoint, and evaluate the model.
        Prints the loss and accuracy.
        """
        model_name = config['model_name']
        models_dir = './models'
        checkpoint_dir = os.path.join(models_dir, model_name, 'checkpoints')

        if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
            raise FileNotFoundError("Checkpoints directory is empty or does not exist!")

        checkpoint_files = sorted(os.listdir(checkpoint_dir))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"Loading model from: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint)
        state_dict = checkpoint['model_state']
        model_state = self.model.state_dict()

        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
        model_state.update(filtered_state_dict)
        self.model.load_state_dict(model_state)

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.dataset:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.dataset)
        avg_accuracy = correct_predictions / total_samples

        print(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4%}")







