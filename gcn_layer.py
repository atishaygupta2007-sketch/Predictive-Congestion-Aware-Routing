import torch
import torch.nn as nn

import numpy as np
import pandas as pd
                                                                         

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, A_norm):
        super().__init__()
        self.A_norm = A_norm
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X):
        X = torch.matmul(self.A_norm, X)
        X = self.linear(X)
        return X




