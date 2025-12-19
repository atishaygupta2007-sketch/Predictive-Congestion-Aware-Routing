import torch
import torch.nn as nn

import numpy as np
import pandas as pd
                                                                         
'''
# -------- Load METR-LA data --------
df = pd.read_hdf("data/METR-LA.h5")   # adjust path if needed
data = df.values                     # shape: (time_steps, 207)

print("Raw data shape:", data.shape)'''


# ---------- GCN Layer ----------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, A_norm):
        super().__init__()
        self.A_norm = A_norm
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X):
        X = torch.matmul(self.A_norm, X)
        X = self.linear(X)
        return X


# ---------- TGCN Model ----------
'''class TGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, pred_len, A_norm):
        super().__init__()

        self.gcn = GCNLayer(1, hidden_dim, A_norm)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, X):
        B, T, N = X.shape

        X = X.permute(0, 2, 1).unsqueeze(-1)

        gcn_out = []
        for t in range(T):
            gcn_out.append(self.gcn(X[:, :, t, :]))

        gcn_out = torch.stack(gcn_out, dim=2)
        gcn_out = gcn_out.view(B * N, T, -1)

        _, h = self.gru(gcn_out)
        h = h.squeeze(0)

        out = self.fc(h)
        out = out.view(B, N, -1).permute(0, 2, 1)
        return out


print("TGCN file runs successfully")

# -------- Create one sample --------
seq_len = 12
pred_len = 3

X = data[:seq_len]                   # (12, 207)
X = torch.tensor(X, dtype=torch.float32)

# Add batch dimension
X = X.unsqueeze(0)                   # (1, 12, 207)

print("Input X shape:", X.shape)


# -------- Load normalized adjacency --------
A_norm = torch.load("data/adj_norm_METR_LA.pt")

print("Adjacency shape:", A_norm.shape)


num_nodes = 207
hidden_dim = 32

model = TGCN(
    num_nodes=num_nodes,
    hidden_dim=hidden_dim,
    pred_len=pred_len,
    A_norm=A_norm
)

# -------- Forward pass --------
with torch.no_grad():
    y_pred = model(X)

print("Output prediction shape:", y_pred.shape)'''
