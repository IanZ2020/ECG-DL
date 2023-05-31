import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.data_utils import EcgDataset
from pycm import *

device = torch.device("cpu")
labels=['N','V','/','L','R']
test_path = 'data/segment_with_beat_sampled/test.csv'
model_path = 'save/5_CNNbidLSTM.pth'
model = torch.load(model_path)
batch_size = 128

testing_data = EcgDataset(file=test_path, labels=labels)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
loss_fn = nn.CrossEntropyLoss()

def evaluate(dataloader, model, loss_fn, device, labels):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = .0, .0
    matrix = {}
    for i in range(len(labels)):
        matrix[labels[i]]={}
        for j in range(len(labels)):
            matrix[labels[i]][labels[j]] = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, torch.squeeze(y,0).type(torch.float)).item()
            y_idx = torch.argmax(y,dim=1)
            y_pred = pred.argmax(dim = 1)
            correct += (y_pred == y_idx).type(torch.float).sum().item()
            for label, pred in zip(y_idx, y_pred):
                matrix[labels[label]][labels[pred]] += 1
    cm = ConfusionMatrix(matrix=matrix)
    cm.print_matrix()
    cm.stat(summary=True)
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

evaluate(test_dataloader, model, loss_fn, device, labels)