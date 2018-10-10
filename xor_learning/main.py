import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
source = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
target = np.array([[0], [1], [1], [0]])

source = torch.from_numpy(source).float()
target = torch.from_numpy(target).float()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

net = Net()

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(0, 1000):

    print("source:", source)
    print('')
    outputs = net(source)
    print("outputs:", outputs)
    loss = criterion(outputs, target)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
