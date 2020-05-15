import torch
from torch import nn
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class LSTMCell(nn.Module):
    def __init__(self, in_features, hide_features):
        super(LSTMCell, self).__init__()
        self.Wi = nn.Linear(in_features=in_features, out_features=hide_features, bias=True)
        self.Wf = nn.Linear(in_features=in_features, out_features=hide_features, bias=True)
        self.Wo = nn.Linear(in_features=in_features, out_features=hide_features, bias=True)
        self.Wc = nn.Linear(in_features=in_features, out_features=hide_features, bias=True)

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


class SineRnn(nn.Module):
    def __init__(self, in_features=1, hide_features=50, out_features=1):
        super(SineRnn, self).__init__()
        self.hide_features = hide_features
        self.lstm_cell1 = LSTMCell(in_features, hide_features)
        self.lstm_cell2 = LSTMCell(hide_features, hide_features)
        self.linear = nn.Linear(in_features=hide_features, out_features=out_features)

    def forward(self, seq, predict=0):
        out = []
        ht_1 = torch.zeros(seq.size(0), self.hide_features, dtype=torch.double).to(device)
        ct_1 = torch.zeros(seq.size(0), self.hide_features, dtype=torch.double).to(device)
        ht_2 = torch.zeros(seq.size(0), self.hide_features, dtype=torch.double).to(device)
        ct_2 = torch.zeros(seq.size(0), self.hide_features, dtype=torch.double).to(device)
        seq = seq.chunk(seq.size(1), dim=1)
        o = None
        for x in seq:
            ht_1, ct_1 = self.lstm_cell1(x, ht_1, ct_1)
            ht_2, ct_2 = self.lstm_cell2(ht_1, ht_2, ct_2)
            o = self.linear(ht_2)
            out += [o]
        for i in range(predict):
            ht_1, ct_1 = self.lstm_cell1(o, ht_1, ct_1)
            ht_2, ct_2 = self.lstm_cell2(ht_1, ht_2, ct_2)
            o = self.linear(ht_2)
            out += [o]
        out = torch.stack(out, 1).squeeze(2)
        return out


class EmotionRnn(nn.Module):
    def __init__(self, in_features=50, hide_features=100, out_features=2):
        super(EmotionRnn, self).__init__()
        self.hide_features = hide_features
        self.lstm_cell1 = LSTMCell(in_features, hide_features)
        self.lstm_cell2 = LSTMCell(hide_features, hide_features)
        self.linear = nn.Linear(in_features=hide_features * 50, out_features=out_features)

    def forward(self, seq):
        seq = seq.squeeze(1)
        ht_1 = torch.zeros(self.hide_features, dtype=torch.double).to(device)
        ct_1 = torch.zeros(self.hide_features, dtype=torch.double).to(device)
        ht_2 = torch.zeros(self.hide_features, dtype=torch.double).to(device)
        ct_2 = torch.zeros(self.hide_features, dtype=torch.double).to(device)
        out = None
        for x in seq:
            ht_1, ct_1 = self.lstm_cell1(x, ht_1, ct_1)
            ht_2, ct_2 = self.lstm_cell2(ht_1, ht_2, ct_2)
            o = self.linear(ht_2)
            if out is None:
                out = ht_2
            else:
                out = torch.cat((out, ht_2), dim=1)

        out = self.linear(out)
        return torch.softmax(out, 1)
