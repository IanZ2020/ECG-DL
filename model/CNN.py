from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
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
            nn.Conv1d(in_channels=32,out_channels=128, kernel_size=5, stride=2),#[32,185]->[128,91]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),#[128,91]->[128,45]

            nn.Conv1d(in_channels=128,out_channels=512, kernel_size=5, stride=1),#[128,45]->[512,41]
            nn.ReLU(),
            nn.Conv1d(in_channels=512,out_channels=128, kernel_size=3, stride=1),#[512,41]->[128,39]
            nn.ReLU(),

            nn.Flatten(-2,-1),#[128,20]->[2560]
            nn.Linear(in_features=4992,out_features=512),#[2560]->[512]
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512,out_features=7),#[512]->[7]
            nn.Softmax(dim=-1)
        )
    def forward(self, input):
        output = self.cnn(input)
        return output