
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

class MSNet(nn.Module):
    def __init__(self):
        super(MSNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding=2),
            nn.SELU()
        )
        self.pool1 = nn.MaxPool2d((4, 1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.SELU()
        )
        self.pool2 = nn.MaxPool2d((4, 1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.SELU()
        )
        self.pool3 = nn.MaxPool2d((4, 1), return_indices=True)
        self.bottom = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 4, 5, padding=(0, 2)),
            nn.SELU()
        )

        self.up_pool3 = nn.MaxUnpool2d((4, 1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.SELU()
        )
        self.up_pool2 = nn.MaxUnpool2d((4, 1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.SELU()
        )

        self.up_pool1 = nn.MaxUnpool2d((4, 1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 4, 5, padding=2),
            nn.SELU()
        )

        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(320, 320)
        self.cc1 = nn.Conv2d(32, 4, 5, padding=2)

        self.fc2 = nn.Linear(80, 320)
        self.cc2 = nn.Conv2d(64, 4, 5, padding=2)

        self.fc3 = nn.Linear(20, 128)
        self.fc3_2 = nn.Linear(128, 320)
        self.cc3 = nn.Conv2d(64, 4, 5, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        out1 = self.fc1(self.selu(self.cc1(x)).permute(0,1,3,2)).permute(0,1,3,2)
        c1, ind1 = self.pool1(x)
        c1 = self.conv2(c1)

        out2 = self.fc2(self.selu(self.cc2(c1)).permute(0,1,3,2)).permute(0,1,3,2)
        c2, ind2 = self.pool2(c1)
        c3, ind3 = self.pool3(self.conv3(c2))
        bm = self.bottom(c3)
        u3 = self.up_conv3(self.up_pool3(c3, ind3))

        out3 = self.fc3_2(self.relu(self.dropout(self.fc3(self.selu(self.cc3(u3)).permute(0,1,3,2))))).permute(0,1,3,2)
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        feat = torch.cat((bm, u1), dim=2)
        output = self.softmax(torch.cat((bm, u1), dim=2))
        out1 = self.softmax(torch.cat((bm, out1), dim=2).squeeze(dim=1))
        out2 = self.softmax(torch.cat((bm, out2), dim=2).squeeze(dim=1))
        out3 = self.softmax(torch.cat((bm, out3), dim=2).squeeze(dim=1))

        return output.squeeze(dim=1), feat, out1, out2, out3

if __name__ == '__main__':
    x = torch.randn(1, 3, 320, 64)
    Net = MSNet()
    print(Net(x)[2].shape)