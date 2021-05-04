import torch
import torch.nn as nn
from torchsummary import summary


class S3D(nn.Module):
    def __init__(self, mini=False):
        '''The size of input data should be 16x112x112 or 16x24x32(depth, height, width)'''
        super().__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0))
        self.se_fc1 = nn.Linear(512, 32)
        self.se_fc2 = nn.Linear(32, 512)
        
        self.fc6 = nn.Linear(512*1*4*3, 4096) if not mini else nn.Linear(512*1*1*1, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, X):
        out = self.relu(self.conv1(X))
        out = self.pool1(out)

        out = self.relu(self.conv2(out))
        out = self.pool2(out)

        out = self.relu(self.conv3a(out))
        out = self.relu(self.conv3b(out))
        out = self.pool3(out)

        out = self.relu(self.conv4a(out))
        out = self.relu(self.conv4b(out))
        out = self.pool4(out)

        out = self.relu(self.conv5a(out))
        out = self.relu(self.conv5b(out))
        out = self.pool5(out)        
        
        out = out.view(-1, torch.prod(torch.FloatTensor(list(out.shape)[1:]), dtype=torch.int))
        out = self.relu(self.fc6(out))
        out = self.dropout(out)
        out = self.relu(self.fc7(out))
        out = self.dropout(out)

        out = self.fc8(out)
        hypothesis = self.sigmoid(out)

        return hypothesis
    

if __name__ == '__main__':  
    # model with 112x112 resolution
    model = S3D().to('cuda')
    summary(model, (1, 16, 112, 112))
    
    # model with 24x32 resolution
    model = S3D(mini=True).to('cuda')
    summary(model, (1, 16, 24, 32))