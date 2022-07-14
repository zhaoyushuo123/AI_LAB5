# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import os
from config import config
import torch.nn.functional as F


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'Gru_Vgg'
        self.train_path = 'new_data/' + dataset + '/train.txt'  # 训练集
        self.dev_path = 'new_data/' + dataset + '/valid.txt'  # 验证集
        self.test_path = 'new_data/' + dataset + '/test.txt'  # 测试集

        self.class_list = [x.strip() for x in open(
            'new_data/' + dataset + '/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = config["vocab_path"]  # 词表
        if not os.path.exists('saved_dict/'): os.mkdir('saved_dict/')
        self.save_path = './saved_dict/' + dataset + '-' + self.model_name + '.ckpt'  # 模型训练结果
        self.embedding_pretrained = torch.tensor(
            np.load('./pretrained/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda:' + config["gpu"] if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = config["dropout"]  # 随机失活
        self.patience = config["patience"]
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = config["epochs"]  # epoch数
        self.batch_size = config["batch_size"]  # mini-batch大小
        self.pad_size = config["pad_size"]  # 每句话处理成的长度(短填长切)
        self.learning_rate = config["learning_rate"]  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数

        self.save_result = config["save_result"]


# gru+vgg16
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.GRU(config.embed, config.hidden_size, config.num_layers,
                           bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc1 = nn.Linear(config.hidden_size * 2, 1)

        # 通道数为3，尺寸为224*244*3，卷积核个数为64，卷积核尺寸为3*3*3,默认步长为1，sample填充（用0填充）
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 输出 64 * 224 * 224
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))  # 64 * 224 * 224
        nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # 输出 64 * 112 * 112
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出128 * 112* 112
        nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1))  # 输出128 * 112* 112
        nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # 输出 128 * 56 * 56
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 256 * 56 * 56
        nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))  # 256 * 56 * 56
        nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))  # 256 * 56 * 56
        nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 512 * 28 * 28
        nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))  # 512 * 28 * 28
        nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))  # 512 * 28 * 28
        nn.ReLU()
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # 512 * 14 * 14
        nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))  # 512 * 14 * 14
        nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))  # 512 * 14 * 14
        nn.ReLU()
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 8 * 8

        # 全连接层
        self.fc2 = nn.Linear(512 * 8 * 8, 4096)
        nn.ReLU()
        # 调用Dropout函数防止过拟合
        nn.Dropout()
        self.fc3 = nn.Linear(4096, 2048)
        nn.ReLU()
        nn.Dropout()
        self.fc4 = nn.Linear(2048 + 1, config.num_classes)

        # softmax 1 * 1 * 1000

    def forward(self, text, image):
        text, _ = text
        text = self.embedding(text)  # [batch_size, seq_len, embeding]=[128, 64, 300]
        text, _ = self.lstm(text)

        text = self.fc1(text[:, -1, :])  # 句子最后时刻的 hidden state

        in_size = image.size(0)
        image = self.conv1_1(image)  # 224
        image = F.relu(image)
        image = self.conv1_2(image)  # 224
        image = F.relu(image)
        image = self.maxpool1(image)  # 112

        image = self.conv2_1(image)  # 112
        image = F.relu(image)
        image = self.conv2_2(image)  # 112
        image = F.relu(image)
        image = self.maxpool2(image)  # 56

        image = self.conv3_1(image)  # 56
        image = F.relu(image)
        image = self.conv3_2(image)  # 56
        image = F.relu(image)
        image = self.conv3_3(image)  # 56
        image = F.relu(image)
        image = self.maxpool3(image)  # 28

        image = self.conv4_1(image)  # 28
        image = F.relu(image)
        image = self.conv4_2(image)  # 28
        image = F.relu(image)
        image = self.conv4_3(image)  # 28
        image = F.relu(image)
        image = self.maxpool4(image)  # 14

        image = self.conv5_1(image)  # 14
        image = F.relu(image)
        image = self.conv5_2(image)  # 14
        image = F.relu(image)
        image = self.conv5_3(image)  # 14
        image = F.relu(image)
        image = self.maxpool5(image)  # 8

        # 展平
        image = image.view(in_size, -1)

        image = self.fc2(image)
        image = F.relu(image)
        image = self.fc3(image)
        image = F.relu(image)
        # image = self.fc3(image)

        out = self.fc4(torch.cat((image, text), 1))

        return out