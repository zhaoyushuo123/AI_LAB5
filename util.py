# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    return vocab_dic


def build_predict_dataset(config, texts):
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    if isinstance(texts, str):
        texts = [texts]

    def load_dataset(texts, pad_size=32):
        contents = []
        for line in texts:
            lin = line.strip()
            if not lin:
                continue
            content = lin
            label = 0
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, label, seq_len))
        return contents

    predict = load_dataset(config.pad_size, texts, )
    config.batch_size = 1
    return predict


def build_dataset(config, ues_word=True):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                print(lin)
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, dev,test


# 文本图像部分数据类
class TextImageDataset(data.Dataset):
    def __init__(self, texts, file_path, device):
        self.data_files = [w.strip() for w in open(file_path, 'r').readlines()]
        self.text_data = texts

        self.class_list = [w.strip() for w in open('new_data/text/class.txt', 'r').readlines()]
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        self.device = device

    def __getitem__(self, item):
        text_data = self.text_data[item]

        text = torch.tensor(text_data[0]).to(self.device)
        text_label = torch.tensor(text_data[1]).to(self.device)
        seq_len = torch.tensor(text_data[2]).to(self.device)

        data_file, image_label = self.data_files[item].split(',')
        print(data_file, " is coming")
        #data_file = os.path.join('new_data/image/data', data_file + '.jpg')
        data_file = 'data/实验五数据/实验五数据/data/'+data_file+'.jpg'
        print(data_file)
        img = Image.open(data_file)


        image_label = self.class_list.index(image_label)

        img = self.transforms(img).to(self.device)
        image_label = torch.tensor(image_label).to(self.device)

        assert text_label.equal(image_label)
        return (text, seq_len), img, image_label

    def __len__(self):
        assert len(self.data_files) == len(self.text_data)

        return len(self.data_files)


def get_iter(text_train, text_valid, text_test,config):
    train_data = TextImageDataset(text_train, 'new_data/image/train.txt', config.device)
    valid_data = TextImageDataset(text_valid, 'new_data/image/valid.txt', config.device)
    test_data = TextImageDataset(text_test, 'new_data/image/test.txt', config.device)

    train_loader = data.DataLoader(train_data, config.batch_size, shuffle=False)
    valid_loader = data.DataLoader(valid_data, config.batch_size, shuffle=False)
    test_loader = data.DataLoader(test_data, config.batch_size, shuffle=False)

    return train_loader, valid_loader,test_loader


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "../data/text/train.txt"
    vocab_dir = "../pretrained/glove/vocab.pkl"
    pretrain_dir = "../pretrained/glove/glove.6B.300d.txt"
    emb_dim = 300
    filename_trimmed_dir = "../pretrained/glove/embedding_glove"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)