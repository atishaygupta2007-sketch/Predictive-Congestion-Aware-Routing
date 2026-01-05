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

results = []
epochs = 1500

for gcn_hidden in [16, 32, 64, 128]:
    for lr in [0.001, 0.0005]:

        print(f"\nTraining with gcn_hidden={gcn_hidden}, lr={lr}")

        model = TGCNModel(
            num_nodes=N,
            in_feat=1,
            gcn_hidden=gcn_hidden,
            seq_len=seq_len,
            pred_len=pred_len,
            A_norm=A_norm
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        final_loss = losses[-1]

        results.append({
            "gcn_hidden": gcn_hidden,
            "lr": lr,
            "final_loss": final_loss
        })

        print(f"Final Loss: {final_loss:.4f}")

        torch.save(model.state_dict(), f"checkpoints/tgcn_gcn{gcn_hidden}_lr{lr}.pth")
        np.save(f"checkpoints/train_losses_gcn{gcn_hidden}_lr{lr}.npy",np.array(losses))


