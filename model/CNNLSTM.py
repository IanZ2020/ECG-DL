from torch import nn

#pick out an element from a tuple or list
#SelectItem can be used in Sequential to pick out the hidden state
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super().__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=128,kernel_size=50, stride=3),#[1,3600]->[128,1184]
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=3),#[128,1184]->[128,395]

            nn.Conv1d(in_channels=128,out_channels=32, kernel_size=7, stride=1),#[128,1184]->[32,389]
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),#[32,389]->[32,194]

            nn.Conv1d(in_channels=32,out_channels=32, kernel_size=10, stride=1),#[32,194]->[32,185]
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),#[32,185]->[32,92]

            nn.LSTM(input_size=92,hidden_size=10,num_layers=1),
            SelectItem(0),#[32,92]->[32,10]

            nn.Flatten(-2,-1),#[32,10]->[320]
            nn.Linear(in_features=320,out_features=20),#[320]->[20]
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=20,out_features=10),#[20]->[10]
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=7),#[10]->[7]
            nn.Softmax()
        )

    def forward(self, input):
        output = self.layers(input)
        return output