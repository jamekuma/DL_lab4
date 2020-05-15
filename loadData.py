import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

def build_dic(fileName):
    dict = {}
    with open(fileName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(" ")
            key = items[0]
            value = [float(item) for item in items[1:]]
            dict[key] = value
    return dict


def get_sentence_seq(sentence, dict):
    '''
    根据一句话，生成其词向量序列。若不满50词则在前面补零；超过50词则取前50个词
    :param sentence: 句子
    :param dict: 词向量字典
    :return: 词向量序列
    '''
    words = sentence.split(' ')
    res = []
    cnt = 0
    for word in words:
        if word in dict.keys():
            res.append(dict[word])
            cnt += 1
        if cnt == 50:
            break

    zeros = [0 for _ in range(50)]

    for i in range(50 - cnt):
        res.insert(0, zeros)
    return res


def get_dataset(pos_file, neg_file, dict, shuffle=True):
    pos_set = []
    neg_set = []
    with open(pos_file, 'r', encoding="windows-1252") as f:
        for line in f.readlines():
            seq = get_sentence_seq(line, dict)
            pos_set.append(seq)
    with open(neg_file, 'r', encoding="windows-1252") as f:
        for line in f.readlines():
            seq = get_sentence_seq(line, dict)
            neg_set.append(seq)

    # 留最后1000句话作为测试集
    train_data = pos_set[:-1000] + neg_set[:-1000]
    train_target = [1 for _ in range(len(pos_set) - 1000)] + [0 for _ in range(len(neg_set) - 1000)]
    test_data = pos_set[-1000:] + neg_set[-1000:]
    test_target = [1 for _ in range(1000)] + [0 for _ in range(1000)]

    # 打乱训练集
    if shuffle:
        permutaion = np.random.permutation(len(train_target))
        train_data_shuffle = np.zeros(np.shape(train_data))
        train_target_shuffle = np.zeros(np.shape(train_target))
        for i in range(len(train_target)):
            train_data_shuffle[i] = np.array(train_data[permutaion[i]])
            train_target_shuffle[i] = np.array(train_target[permutaion[i]])
        
        # print(np.array(train_data_shuffle).shape)
        # print(np.array(train_target_shuffle).shape)
        # print(np.array(test_data).shape)
        # print(np.array(test_target).shape)
        return train_data_shuffle, train_target_shuffle.reshape(train_target_shuffle.shape[0]), np.array(test_data), np.array(test_target).reshape(2000)
    else:
        return train_data, train_target, test_data, test_target

class MyEmotonSet(Dataset):
    def __init__(self, root_path, train=True):
        super(MyEmotonSet, self).__init__()
        dict = build_dic(root_path + 'glove.6B.50d.txt')
        train_data, train_target, test_data, test_target = get_dataset(
            root_path + 'rt-polarity.pos',
            root_path + 'rt-polarity.neg',
            dict)
        self.transform =  transforms.Compose([transforms.ToTensor()])
        if train:
            self.datas = train_data
            self.targets = torch.LongTensor(train_target)
        else:
            self.datas = test_data
            self.targets = torch.LongTensor(test_target)

    def __getitem__(self, index):
        data = self.datas[index]
        target = self.targets[index]
        data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.datas)