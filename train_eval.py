# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import f1_score
import time
from util import get_time_dif
from log import logger
from tqdm import tqdm
from transformers import AdamW
import os
from torch.optim import lr_scheduler


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    dev_best_f1 = float('-inf')

    last_improve_epoch = 0
    model.train()

    for epoch in range(config.num_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        exp_lr_scheduler.step()
        # 记录变量
        train_labels_all = np.array([], dtype=int)
        train_predicts_all = np.array([], dtype=int)
        train_loss_list = []

        for text_trains, image_trains, labels in tqdm(train_iter):
            model.train()
            model.zero_grad()

            outputs = model(text_trains, image_trains)

            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            train_loss_list.append(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪

            # 真实标签和预测标签
            labels = labels.data.cpu()
            predicts = torch.max(outputs.data, 1)[1].cpu()

            train_labels_all = np.append(train_labels_all, labels)
            train_predicts_all = np.append(train_predicts_all, predicts)

        # 训练集评估
        train_loss = sum(train_loss_list) / (len(train_loss_list) + 1e-10)
        train_acc = metrics.accuracy_score(train_labels_all, train_predicts_all)
        train_f1 = metrics.f1_score(train_labels_all, train_predicts_all, average='macro')

        dev_acc, dev_f1, dev_loss, report, confusion = evaluate(config, model, dev_iter)
        if config.save_result:
            if not os.path.exists(f'result'):
                os.mkdir(f'result')
            if not os.path.exists(f'result/{config.model_name}'):
                os.mkdir(f'result/{config.model_name}')
            np.save(f'result/{config.model_name}/{epoch}.npy',
                    np.array([train_loss, train_f1, train_acc, dev_loss, dev_f1, dev_acc]))
        optimizer.step()
        msg = 'Train Loss: {0:>5.6},  Train Acc: {1:>6.2%},  Train F1: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Val F1: {5:>6.2%}'
        logger.info(msg.format(train_loss, train_acc, train_f1, dev_loss, dev_acc, dev_f1))
        logger.info("Precision, Recall and F1-Score...")
        logger.info(report)
        logger.info("Confusion Matrix...")
        logger.info(confusion)

        if dev_f1 > dev_best_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), config.save_path)
            last_improve_epoch = epoch

        if epoch - last_improve_epoch > config.patience:
            logger.info("No optimization for a long time, auto-stopping...")
            break

    test(config, model, dev_iter)
    test_result = predict(config, model, test_iter)
    test_result = [config.class_list[w] for w in test_result]
    test_id = [w.strip().split(",")[0] for w in open('data/实验五数据/test_without_label.txt', 'r').readlines()][1:]
    with open('data/实验五数据/test_with_label.txt','w') as f:
        for test_id,test_data in zip(test_id,test_result):
            f.write(test_id+','+test_data+'\n')


def test(config, model, dev_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_f1, test_loss, test_report, test_confusion = evaluate(config, model, dev_iter)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1:{2:>6.2%}'
    logger.info(msg.format(test_loss, test_acc, test_f1))
    logger.info("Precision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("Confusion Matrix...")
    logger.info(test_confusion)
    time_dif = get_time_dif(start_time)
    logger.info(f"Time usage:{time_dif}")


def evaluate(config, model, dev_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for text_trains, image_trains, labels in tqdm(dev_iter):
            outputs = model(text_trains, image_trains)

            # print(labels)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all, average='macro')
    report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)

    return acc, f1, loss_total / (len(dev_iter) + 1e-10), report, confusion


def predict(config, model, data_iter):
    model.eval()
    model.load_state_dict(torch.load(config.save_path))
    predict_all = np.array([], dtype=int)

    with torch.no_grad():
        for text_trains, image_trains, labels in tqdm(data_iter):
            outputs = model(text_trains, image_trains)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)

    return predict_all.tolist()