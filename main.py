import torch
import numpy as np

import torch.nn as nn
from models.tgcn_model import TGCNModel

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

A_norm = torch.load("data/adj_norm_METR_LA.pt")

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return (torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float())

df = pd.read_hdf("data/METR-LA.h5")
data = (df.values-df.values.mean())/df.values.std()   # (time, 207)

seq_len = 12
pred_len = 3
B = 32
N = 207

X, y = create_sequences(data, seq_len, pred_len)

X = X[:32]   
y = y[:32]   

dataset = TensorDataset (X, y)
loader = DataLoader (dataset, batch_size = 32, shuffle = True)

model = TGCNModel(
    num_nodes=N,
    in_feat=1,
    gcn_hidden=16,
    seq_len=seq_len,
    pred_len=pred_len,
    A_norm=A_norm
)

y_pred = model(X)
print("Output shape:", y_pred.shape)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 500
model.train()

train_losses = []

for epoch in range(epochs):
    total_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "checkpoints/tgcn_metrla.pth")
np.save("checkpoints/train_losses.npy",np.array(train_losses))