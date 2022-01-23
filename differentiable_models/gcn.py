import torch
import torch.nn as nn
from deeprobust.graph import utils


# differentiable
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, dropout=0, bias=True):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(in_ft, 2*out_ft, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(2*out_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias1 = nn.Parameter(torch.FloatTensor(2*out_ft))
            self.bias1.data.fill_(0.0)
            self.bias2 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias2.data.fill_(0.0)
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (nodes, features)
    def forward(self, seq, adj, sparse=False):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=sparse)
        seq_fts1 = self.fc1(seq)
        if sparse:
            out1 = torch.spmm(adj_norm, seq_fts1)
        else:
            out1 = torch.mm(adj_norm, seq_fts1)
        if self.bias1 is not None:
            out1 += self.bias1
        out1 = self.act(out1)
        out1 = self.dropout(out1)

        seq_fts2 = self.fc2(out1)
        if sparse:
            out2 = torch.spmm(adj_norm, seq_fts2)
        else:
            out2 = torch.mm(adj_norm, seq_fts2)
        if self.bias2 is not None:
            out2 += self.bias2
        return self.act(out2)
