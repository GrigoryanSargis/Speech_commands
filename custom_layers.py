import torch
import torch.nn as nn

class DataNormalization(nn.Module):

    def __init__(self):
        super(DataNormalization, self).__init__()
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.momentum = 0.1  # Momentum for updating mean and variance

    def forward(self, x):

        if self.training:
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(dim=0, unbiased=False, keepdim=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            return (x - batch_mean) / (torch.sqrt(batch_var + 1e-6))
        else:
            return (x - self.running_mean) / (torch.sqrt(self.running_var + 1e-6))

class FeatureLayer(nn.Module):
    def __init__(self, feature_extractor):
        super(FeatureLayer, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features