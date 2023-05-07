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
            nn.Conv1d(in_channels=1,out_channels=10,kernel_size=3, stride=1),#[1,3600]->[10,3598]
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(kernel_size=2, stride=2),#[10,3598]->[10,1799]

            nn.Conv1d(in_channels=10,out_channels=100, kernel_size=4, stride=1),#[10,1799]->[100,1796]
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.MaxPool1d(kernel_size=2, stride=2),#[100,1796]->[100,898]

            nn.Conv1d(in_channels=100,out_channels=200, kernel_size=4, stride=1),#[100,898]->[200,895]
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.MaxPool1d(kernel_size=2, stride=2),#[200,895]->[200,447]

            nn.LSTM(input_size=447,hidden_size=100,num_layers=1),
            SelectItem(0),#[200,447]->[200,100]

            nn.Flatten(-2,-1),#[200,100]->[20000]
            nn.Linear(in_features=20000,out_features=300),#[20000]->[300]
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=300,out_features=20),#[300]->[20]
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=20,out_features=7),#[20]->[7]
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        output = self.layers(input)
        return output