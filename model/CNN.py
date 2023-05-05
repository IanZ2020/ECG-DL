from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(out_channels=128,kernel_size=50, stride=3, padding='same'),
            nn.RelU(),
            nn.BatchNorm1d(),
            nn.MaxPool1d(kernel_size=2, strides=3),

            nn.Conv1d(out_channels=32, kernel_size=7, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.MaxPool1d(kernel_size=2, strides=2),

            nn.Conv1d(out_channels=32, kernel_size=10, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(out_channels=128, kernel_size=5, stride=2, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, strides=2),

            nn.Conv1d(out_channels=512, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(out_channels=128, kernel_size=3, stride=2, padding='same'),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_features=7),
            nn.Softmax()
        )
    def foward(self, input):
        output = self.cnn(input)
        return output