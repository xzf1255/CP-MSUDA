import torch.nn as nn
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.nor = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        # x = self.nor(x)
        return x