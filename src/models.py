import torch
from torch_geometric.nn import GCNConv

class GCN4(torch.nn.Module):
    def __init__(self, in_dim, hidden=32, num_layers=4, out_dim=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_dim))

    def forward(self, x, edge_index, return_io=False):
        inputs, outputs = [], []   # NEW
        for i, conv in enumerate(self.convs):
            inputs.append(x)       # guarda entrada antes de la capa
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
            outputs.append(x)      # guarda salida
        if return_io:
            return x, inputs, outputs
        return x