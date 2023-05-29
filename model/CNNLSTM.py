from torch import nn
import torch
#pick out an element from a tuple or list
#SelectItem can be used in Sequential to pick out the hidden state
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super().__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = 'transpose'
    def forward(self, inputs):
        return torch.transpose(inputs, -2, -1)
class CNNLSTM(nn.Module):
    def __init__(self, bidirectional = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=5,kernel_size=3, stride=1),#[1,360]->[5,358]
            nn.ReLU(),
            nn.BatchNorm1d(5),
            nn.MaxPool1d(kernel_size=2, stride=2),#[5,358]->[5,179]

            nn.Conv1d(in_channels=5,out_channels=10, kernel_size=4, stride=1),#[5,179]->[10, 176]
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(kernel_size=2, stride=2),#[10, 176]->[10, 88]

            Transpose(),#[10, 88]->[88,10]
            nn.LSTM(input_size=10,hidden_size=64,num_layers=1,batch_first=True, bidirectional = self.bidirectional),
            SelectItem(0),#[88,level+1]->[88,64]
            nn.Dropout(p=0.1),
            nn.LSTM(input_size=64,hidden_size=32,num_layers=1,batch_first=True, bidirectional = self.bidirectional),
            SelectItem(0),#[88,64]->[88,32]
            nn.Dropout(p=0.1),

            nn.Flatten(-2,-1),#[88,32]->[2816]
            nn.Linear(in_features=2816,out_features=128),#[2816]->[128]
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128,out_features=5),#[128]->[5]
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        output = self.layers(input)
        return output
    
# x = torch.rand(5,1,360)
# model = CNNLSTM()
# pred = model(x)
# print(pred)