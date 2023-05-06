import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from model.CNN import CNN
from model.CNNLSTM import CNNLSTM

labels=['N','V','L','R','A','F','S']
labels_map = {label: i for i, label in enumerate(labels)}

#hypermeter
epochs = 20
batch_size = 32
learning_rate = 0.001

#file path
train_path = 'data/segmented_data/mitarr_train.csv'
test_path = 'data/segmented_data/mitarr_test.csv'


class EcgDataset(Dataset):
    def __init__(self, file):
        self.data = pd.read_csv(file)
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data.iloc[index,0:len(self.data)]
        y = self.data.iloc[index,len(self.data)]
        x = torch.tensor(x,dtype=torch.float)
        y_one_hot = nn.functional.one_hot(torch.tensor(labels_map[y]), len(labels))
        return x.unsqueeze(0), y_one_hot

training_data = EcgDataset(file=train_path)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_data = EcgDataset(file=test_path)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x)
        y = torch.squeeze(y,0).type(torch.float)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, torch.squeeze(y,0).type(torch.float)).item()
            y_idx = torch.argmax(y,dim=1)
            correct += (pred.argmax(dim = 1) == y_idx).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

print("Saving model")
torch.save(model, 'model.pth')