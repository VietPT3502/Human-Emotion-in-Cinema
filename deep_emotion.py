import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.fundCNN = FundCNN(filters=[11, 9, 11, 8, 11, 7, 11, 27])
        self.identity1 = nn.Conv2d(in_channels=11*3, out_channels= 27, kernel_size=1, stride=2)
        self.cnnBlock1 = CNNBlock(filters=[27, 19, 27, 26, 27, 36], externals=[2, 6],
                                  is_pool=False, id_filter=27, in_channels=27)
        self.identity2 = nn.Conv2d(in_channels= 27*4, out_channels=64, kernel_size=1, stride=2)
        self.cnnBlock2 = CNNBlock(filters=[64, 39, 64, 24, 64], externals=[2, 5, 6],
                                  is_pool=True, id_filter=64, in_channels=36)
        self.fc = nn.Linear(in_features= 256, out_features=7)

    def forward(self, x):
        b0, b0_1, b0_3, b0_5 = self.fundCNN(x)
        out = self.identity1(torch.cat([b0_1, b0_3, b0_5], 1))
        b1, b1_3, b1_5 = self.cnnBlock1(b0, out)
        out2 = self.identity2(torch.cat([b1_3, b1_5, out, b0], 1))
        b2, _, _ = self.cnnBlock2(b1, out2)
        b2 = b2.reshape(b2.shape[0], -1)
        fc = self.fc(b2)
        return fc



class FundCNN(nn.Module):
    def __init__(self, filters):
        super(FundCNN, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Conv2d(1, filters[0], 3, padding='same')
        self.layer2 = nn.Conv2d(filters[0], filters[1], 3, padding='same')
        self.layer3 = nn.Conv2d(filters[1], filters[2], 3, padding='same')
        self.layer4 = nn.Conv2d(filters[0] + filters[2], filters[3], 3,  padding='same')
        self.layer5 = nn.Conv2d(filters[3], filters[4], 3,  padding='same')
        self.layer6 = nn.Conv2d(filters[0] + filters[2] + filters[4], filters[5], 3,  padding='same')
        self.layer7 = nn.Conv2d(filters[5], filters[6], 3, padding='same')
        self.layer8 = nn.Conv2d(filters[0] + filters[4] + filters[6], filters[7], 3,  padding=1, stride=2)

    def forward(self, x):
        out_1 = self.relu(self.layer1(x))
        out_2 = self.relu(self.layer2(out_1))
        out_3 = self.relu(self.layer3(out_2))
        out_4 = self.relu(self.layer4(torch.cat([out_1, out_3], 1)))
        out_5 = self.relu(self.layer5(out_4))
        out_6 = self.relu(self.layer6(torch.cat([out_1, out_3, out_5], 1)))
        out_7 = self.relu(self.layer7(out_6))
        out_8 = self.relu(self.layer8(torch.cat([out_1, out_5, out_7], 1)))
        return out_8, out_1, out_3, out_5

# model = FundCNN(filters=[11, 9, 11, 8, 11, 7, 11, 27])
# x = torch.randn(4, 1, 48, 48)
# out_8, out_1, out_3, out_5 = model(x)
# print(out_1.shape)
# print(out_3.shape)
# print(out_5.shape)
# print(out_8.shape)

class CNNBlock(nn.Module):
    def __init__(self, filters, externals, is_pool, id_filter, in_channels):
        super(CNNBlock, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Conv2d(in_channels, filters[0], 3, padding='same')
        self.layer2 = nn.Conv2d(filters[0] + id_filter, filters[1], 3, padding='same')
        self.layer3 = nn.Conv2d(filters[1] , filters[2], 3, padding='same')
        if 5 in externals:
            self.layer4 = nn.Conv2d(filters[0] + filters[2] + id_filter, filters[3], 3, padding='same')
        else:
            self.layer4 = nn.Conv2d(filters[0] + filters[2], filters[3], 3, padding='same')
        self.layer5 = nn.Conv2d(filters[3], filters[4], 3, padding='same')
        if is_pool:
            self.layer6 = nn.AvgPool2d(kernel_size=12)
        else:
            self.layer6 = nn.Conv2d(filters[0] + filters[2] + filters[4] + id_filter, filters[5], 3, padding=1, stride=2)
        self.ext = externals
        self.is_pool = is_pool

    def forward(self, x, externals):
        out_1 = self.relu(self.layer1(x))
        out_2 = self.relu(self.layer2(torch.cat([out_1, externals], 1)))
        out_3 = self.relu(self.layer3(out_2))
        if 5 in self.ext:
            out_4 = self.relu(self.layer4(torch.cat([out_3, out_1, externals], 1)))
        else:
            out_4 = self.relu(self.layer4(torch.cat([out_3, out_1], 1)))
        out_5 = self.relu(self.layer5(out_4))
        out_6 = self.layer6(torch.cat([out_3, out_5, out_1, externals], 1))
        if not self.is_pool:
            out_6 = self.relu(out_6)


        return out_6, out_3, out_5

