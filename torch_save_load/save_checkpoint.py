import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print('net:')
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4


print('torch.save:')

print(
    'saved var type:',
    type(EPOCH),
    type(net.state_dict()),
    type(optimizer.state_dict()),
    type(LOSS),
)

torch.save(
    {
        'epoch': EPOCH,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': LOSS,
    },
    PATH)
