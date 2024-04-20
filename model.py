import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassification(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassification, self).__init__()
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()
