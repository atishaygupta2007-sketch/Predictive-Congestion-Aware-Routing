import torch
import torch.nn as nn
from .gcn_layer import GCNLayer


class TGCNModel(nn.Module):
    def __init__(self, num_nodes, in_feat, gcn_hidden, seq_len, pred_len, A_norm):
        super().__init__()

        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.gcn1 = GCNLayer(in_feat, gcn_hidden, A_norm)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden, A_norm)

        self.gru = nn.GRU(input_size = gcn_hidden, hidden_size = gcn_hidden, batch_first = True)
        self.fc = nn.Linear(gcn_hidden, pred_len)


    def forward(self, x):
           
        B, T, N = x.shape 

        x = x.permute(0, 2, 1)

        outs = []
        for t in range(self.seq_len):
            xt = x[:, :, t]          
            xt = xt.unsqueeze(-1)   
            xt = self.gcn1(xt)      
            xt = torch.relu(xt)

            xt = self.gcn2(xt)

            outs.append(xt)

        x = torch.stack(outs, dim=2)

        # collapse features
        x = x.view(B*N, T, -1)
        _, h = self.gru(x)
        h = h.squeeze(0)
        out = self.fc(h)
        out = out.view(B, N, self.pred_len)

        return out.permute(0, 2, 1)