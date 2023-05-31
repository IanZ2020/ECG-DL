import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.CNN import CNN
from model.CNNLSTM import CNNLSTM
from model.WTLSTM import WTLSTM
import wandb
from utils.data_utils import EcgDataset

device = torch.device("cpu")

labels=['N','V','/','L','R']

#hypermeter
model_name = "WTLSTM"
epochs = 80
batch_size = 128
learning_rate = 0.0001
step_size = 5
bidirectional = True

#file path
train_path = 'data/small_data/train.csv'
test_path = 'data/small_data/test.csv'

#model
model = WTLSTM(bidirectional = bidirectional,level=3)

training_data = EcgDataset(file=train_path, labels=labels)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_data = EcgDataset(file=test_path, labels=labels)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
loss_fn = nn.CrossEntropyLoss()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x, y = x.to(device), y.to(device)
        pred = model(x)
        y = torch.squeeze(y,0).type(torch.float)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # count corrects
        y_idx = torch.argmax(y,dim=1)
        correct += (pred.argmax(dim = 1) == y_idx).type(torch.float).sum().item()
        if (batch+1) % step_size == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            correct /= (step_size-1) * batch_size + len(x)
            print(f"Train Accuracy: {(100*correct):>0.1f}%, Train Loss: {loss:>8f}   [{current:>5d}/{size:>5d}]")
            wandb.log({"train_acc": correct, "train_loss": loss})
            correct = .0

def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = .0, .0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, torch.squeeze(y,0).type(torch.float)).item()
            y_idx = torch.argmax(y,dim=1)
            correct += (pred.argmax(dim = 1) == y_idx).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    wandb.log({"test_acc": correct, "test_loss": test_loss})
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

wandb.init(
    # set the wandb project where this run will be logged
    project="ecg-"+model_name,
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": model_name,
    "dataset": train_path,
    "epochs": epochs,
    "batch_size": batch_size,
    "step_size": step_size
    }
)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device, batch_size)
    test_loop(test_dataloader, model, loss_fn, device)
wandb.finish()
print("Done!")

print("Saving model")
model_path = model_name + f"_bid_{learning_rate}lr_{epochs}epochs_{batch_size}bs_{step_size}step_size.pth"
torch.save(model, model_path)