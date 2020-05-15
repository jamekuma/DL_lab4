import torch
from torch import nn
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class LSTMCell(nn.Module):
    def __init__(self, in_features, hide_features, out_features):
        super(LSTMCell, self).__init__()
        self.Wi = nn.Linear(in_features=in_features, out_features=hide_features, bias=False)
        self.Wf = nn.Linear(in_features=in_features, out_features=hide_features, bias=False)
        self.Wo = nn.Linear(in_features=in_features, out_features=hide_features, bias=False)
        self.Wc = nn.Linear(in_features=in_features, out_features=hide_features, bias=False)

        self.Ui = nn.Linear(in_features=hide_features, out_features=hide_features, bias=True)
        self.Uf = nn.Linear(in_features=hide_features, out_features=hide_features, bias=True)
        self.Uo = nn.Linear(in_features=hide_features, out_features=hide_features, bias=True)
        self.Uc = nn.Linear(in_features=hide_features, out_features=hide_features, bias=True)

    def forward(self, x, h, c):
        # print(x.size())
        # print(h.size())
        it = torch.sigmoid(self.Wi(x) + self.Ui(h))
        ft = torch.sigmoid(self.Wf(x) + self.Uf(h))
        ot = torch.sigmoid(self.Wo(x) + self.Uo(h))
        _ct = torch.tanh(self.Wc(x) + self.Uc(h))
        ct = ft * c + it * _ct
        ht = ot * torch.tanh(ct)
        return ht, ct


class LSTMRnn(nn.Module):
    def __init__(self, in_features=1, hide_features=50, out_features=1):
        super(LSTMRnn, self).__init__()
        self.h0 = torch.zeros(hide_features, dtype=torch.double).to(device)
        self.c0 = torch.zeros(out_features, dtype=torch.double).to(device)
        self.lstm_cell = LSTMCell(in_features, hide_features, out_features)
        self.linear = nn.Linear(in_features=hide_features, out_features=out_features)

    def forward(self, seq, predict=0):
        seq = seq.chunk(seq.size(1), dim=1)
        out = []
        h = self.h0
        c = self.c0
        o = None
        for x in seq:
            ht, ct = self.lstm_cell(x, h, c)
            h = ht
            c = ct
            o = self.linear(h)
            out += [o]
        for i in range(predict):
            ht, ct = self.lstm_cell(o, h, c)
            h = ht
            c = ct
            o = self.linear(h)
            out += [o]
        out = torch.stack(out, 1).squeeze(2)
        return out
