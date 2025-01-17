import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers import DataNormalization


class My_Model(nn.Module):
    def __init__(self, num_classes=35, input_features=13, lstm_hidden_size=256, lstm_layers=2, dropout=0.3):
        super(My_Model, self).__init__()

        self.data_norm = DataNormalization()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTM(
            input_size=128 * (input_features // 4), 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(2 * lstm_hidden_size, 256) 
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.data_norm(x)

        x = x.unsqueeze(1) if x.dim() == 3 else x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, width, -1)  

        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

"""

Here you should implement a network 
It should be LSTM or convolutional
You can implement any thing if you can reach accuracy >85% 
It should be tf.keras.Model
you are free to use ani API
"""
