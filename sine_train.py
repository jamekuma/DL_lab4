import torch
import argparse
from LSTM import LSTMRnn
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='learn_rate', default=0.1, type=int)
parser.add_argument('--epochs', dest='epochs', default=100, type=int)
args = parser.parse_args()
learn_rate = args.learn_rate
epochs = args.epochs

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 画线
def draw(y, test, i):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks()
    plt.yticks()
    plt.plot(np.arange(999), test[0][:999], linewidth=2.0, color = 'blue')
    plt.plot(np.arange(999), test[1][:999], linewidth=2.0, color = 'yellow')
    plt.plot(np.arange(999), test[2][:999], linewidth=2.0, color = 'red')
    plt.plot(np.arange(999, 1999), y[0][999:], linestyle=':', linewidth=2.0, color = 'blue')
    plt.plot(np.arange(999, 1999), y[1][999:], linestyle=':', linewidth=2.0, color = 'yellow')
    plt.plot(np.arange(999, 1999), y[2][999:], linestyle=':', linewidth=2.0, color = 'red')
    print(i)
    plt.savefig('predict' + str(i) + '.jpg' )
    plt.close()


# 反向传播，利用 LBFGS 优化器进行优化
def closure():
    optimizer.zero_grad()
    out = model(input)
    loss = lossfunc(out, target).to(device)
    print('loss:', loss.item())
    loss.backward()
    return loss


def train():
    for epoch in range(epochs):
        optimizer.step(closure)
        test(epoch)


def test(id):
    with torch.no_grad():
        predict = 1000
        pred = model(test_input, predict=predict)
        loss = lossfunc(pred[:, :-predict], test_target)
        print('test loss:', loss.item())
        y = pred.detach().cpu().numpy()
    draw(y, test_target.cpu().numpy(), id)


data = torch.load('./traindata.pt')
input = torch.from_numpy(data[3:, :-1]).to(device)
target = torch.from_numpy(data[3:, 1:]).to(device)
test_input = torch.from_numpy(data[:3, :-1]).to(device)
test_target = torch.from_numpy(data[:3, 1:]).to(device)

lossfunc = nn.MSELoss()
model = LSTMRnn(in_features=1, hide_features=20, out_features=1).to(device).double()
optimizer = torch.optim.LBFGS(model.parameters(), lr=learn_rate)
train_data = torch.from_numpy(torch.load('traindata.pt')).to(device)
train()
model.eval()


