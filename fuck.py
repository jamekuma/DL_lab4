import torch
import torch.nn as nn

t = torch.Tensor([0.10819513453010837]).cuda()
print(t)
layer = nn.Linear(1, 10).cuda()
print(layer(t))
