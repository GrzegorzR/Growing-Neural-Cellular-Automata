import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

class CAModel2(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_chanels= 16, hidden_size=128):
        super(CAModel2, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.hidden_chanels = hidden_chanels
        #self.conv_w = 1
        #dx = np.outer([1, 2, 1], [-1, 0, 1])
        #dy = dx.T
        #conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
        #conv_weights = conv_weights.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
        self.conv = nn.Conv2d(16, hidden_chanels, kernel_size=(3,3), padding=1)
        #print(self.conv.weight.shape)
        self.drop = nn.Dropout()
        self.fc0 = nn.Linear(hidden_chanels, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=True)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        a = x[:, 4:5, :, :] > 0.05
        #print(x[:, 4, 50:60, 50:60])
        return a

    def update(self, x):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.conv(x)
        #print(self.conv.weight)
        #print(dx.shape)
        #dx = F.relu(dx)
        dx = torch.reshape(dx, (x.shape[2]*x.shape[2]*x.shape[0],self.hidden_chanels))
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        #dx = self.drop(dx)
        #dx = F.relu(dx)
        #print(dx.shape)
        dx = torch.reshape(dx, (x.shape[0],  self.channel_n, x.shape[2], x.shape[2]))


        x = x+dx

        post_life_mask = self.alive(dx)

        life_mask = (pre_life_mask & post_life_mask).float()
        #print()
        x = x * post_life_mask.float()

        #print(x)
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x)
        return x

if __name__ == '__main__':
    ca_m = CAModel2(16, 0.2,'cpu')
    ca_m.eval()
    arr = np.random.random((8, 112, 112, 16))
    output = ca_m(torch.from_numpy(arr.astype(np.float32)), 1)
