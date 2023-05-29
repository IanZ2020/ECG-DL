import torch
from torch import nn
import pywt
import numpy as np

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super().__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class WTLayer(nn.Module):
    def __init__(self, wavelet='db6', level=2):
        super(WTLayer, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        coeffs = []
        batch_size = x.size(0)

        for i in range(batch_size):
            signal = x[i,0].cpu().numpy()
            max_length = len(signal)
            c = pywt.wavedec(signal, wavelet=self.wavelet, level=self.level)
            c[0] = signal
            padded_c = torch.zeros((len(c), max_length), dtype=torch.float32)
            for i, row in enumerate(c):
                padded_c[i, :len(row)] = torch.from_numpy(row)
            coeffs.append(torch.transpose(padded_c, 0, 1))

        return torch.stack(coeffs)
    
class WTLSTM(nn.Module):
    def __init__(self, wavelet='db6',level=2, bidirectional = False):
        super().__init__()
        self.level = level
        self.wavelet = wavelet
        self.bidirectional = bidirectional
        self.layers = nn.Sequential(
            WTLayer(wavelet=self.wavelet, level=self.level),

            nn.LSTM(input_size=self.level+1,hidden_size=64,num_layers=1,batch_first=True, bidirectional = self.bidirectional),
            SelectItem(0),#[360,level+1]->[360,64]
            nn.Dropout(p=0.1),
            nn.LSTM(input_size=64,hidden_size=32,num_layers=1,batch_first=True, bidirectional = self.bidirectional),
            SelectItem(0),#[360,64]->[360,32]
            nn.Dropout(p=0.1),

            nn.Flatten(-2,-1),#[360,32]->[11520]
            nn.Linear(in_features=11520,out_features=128),#[11520]->[128]
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128,out_features=5),#[128]->[5]
            nn.Softmax(dim=-1)
        )
    def forward(self, input):
        output = self.layers(input)
        return output
    
# x = torch.rand(21,360)
# model = WTLSTM()
# pred=model(x)
# print(pred)