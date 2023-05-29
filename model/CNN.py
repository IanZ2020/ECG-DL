from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=5,kernel_size=3, stride=1),#[1,360]->[5,358]
            nn.ReLU(),
            nn.BatchNorm1d(5),
            nn.MaxPool1d(kernel_size=2, stride=2),#[5,358]->[5,179]

            nn.Conv1d(in_channels=5,out_channels=10, kernel_size=4, stride=1),#[5,179]->[10, 176]
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(kernel_size=2, stride=2),#[10, 176]->[10, 88]

            nn.Conv1d(in_channels=10,out_channels=20, kernel_size=4, stride=1),#[10, 88]->[20,85]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),#[20,85]->[20,42]

            nn.Flatten(-2,-1),#[20,43]->[840]
            nn.Linear(in_features=840,out_features=30),#[840]->[30]
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=30,out_features=20),#[30]->[20]
            nn.ReLU(),
            nn.Linear(in_features=20,out_features=5),#[20]->[5]
            nn.Softmax(dim=-1)
        )
    def forward(self, input):
        output = self.cnn(input)
        return output