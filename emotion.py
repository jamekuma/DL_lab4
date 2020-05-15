import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from Nets import EmotionRnn
from loadData import MyEmotonSet
import torch.utils.data

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train(model, train_dataset, test_dataset, log_name, n_epochs, batch_size, learning_rate):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    # 定义损失函数和优化器
    writer = SummaryWriter(f'./emotion_log/' + log_name + '/')
    lossfunc = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
    # 开始训练
    for epoch in range(n_epochs):
        if (epoch + 1) % 30 == 0:   # 每30个epoch, 学习率乘0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            # print('data: ')
            # print(data.size())
            target = target.to(device)
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 得到预测值
            # print(target.size())
            # print(output.size())
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('\tEpoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        # train_accuracy = test(model, train_loader) # 训练集准确率
        # print('\ttrain Accuracy: %.4f %%' % (100 * train_accuracy))
        test_accuracy = test(model, test_dataset, batch_size)  # 测试集准确率
        print('\ttest Accuracy: %.4f %%' % (100 * test_accuracy))

        writer.add_scalar('train/Loss', train_loss, epoch + 1)
        # writer.add_scalar('train/Accuracy', train_accuracy, epoch + 1)
        writer.add_scalar('test/Accuracy', test_accuracy, epoch + 1)
        writer.flush()
    writer.close()


# 在数据集上测试神经网络
def test(model, test_dataset, batch_size):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集中不需要反向传播
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)   # 概率最大的就是输出的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 有几个相同的

    return correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", dest="batch_size", default=128, type=int)
    parser.add_argument("--epochs", dest="epochs", default=100, type=int)
    parser.add_argument("--learning_rate", dest="lr", default=0.01, type=float)
    args = parser.parse_args()
    model = EmotionRnn().to(device).double()
    train_set = MyEmotonSet(root_path='./', train=True)
    test_set = MyEmotonSet(root_path='./', train=False)
    train(model, train_set, test_set, log_name='Emotion', n_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)