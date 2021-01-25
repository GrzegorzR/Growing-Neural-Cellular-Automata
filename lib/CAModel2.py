import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

class CAModel2(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(CAModel2, self).__init__()
        self.device = device
        self.channel_n = channel_n

        #conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
        #conv_weights = conv_weights.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=self.channel_n)
        self.conv2 =nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=self.channel_n)

        self.conv1.weight = nn.Parameter(torch.from_numpy(dx.astype(np.float32)).view(1,1,3,3).repeat(self.channel_n, 1, 1, 1))
        self.conv2.weight = nn.Parameter(torch.from_numpy(dy.astype(np.float32)).view(1,1,3,3).repeat(self.channel_n, 1, 1, 1))

        #self.conv = nn.Conv2d(16, hidden_chanels, kernel_size=(3,3), padding=1)
        #print(self.conv.weight.shape)
        self.drop = nn.Dropout(0.5)
        self.fc0 = nn.Linear(48, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=True)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y = torch.cat((x,y1,y2),1)
        return y


    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def update(self, x):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x)
        #print(self.conv1.weight)
        #print(dx.shape)
        #dx = F.relu(dx)
        dx = torch.reshape(dx, (x.shape[2]*x.shape[2]*x.shape[0], 48))

        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        dx = self.drop(dx)
        #dx = F.relu(dx)
        #print(dx.shape)
        dx = torch.reshape(dx, (x.shape[0],  self.channel_n, x.shape[2], x.shape[2]))


        x = x+dx

        post_life_mask = self.alive(dx)

        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask.float()


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
