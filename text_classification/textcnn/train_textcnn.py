from __future__ import absolute_import, division, print_function
import sys
sys.path.append("..")
from typing import Tuple

import oneflow as flow
import oneflow.typing as tp
import argparse
import numpy as np
import os
import shutil
import json
import time
from textcnn import TextCNN
from utils import pad_sequences, load_imdb_data

parser = argparse.ArgumentParser()
parser.add_argument('--ksize_list', type=str, default='2,3,4,5')
parser.add_argument('--n_filters', type=int, default=100)
parser.add_argument('--emb_dim', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--sequence_length', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_load_dir', type=str, default='')
parser.add_argument('--model_save_every_n_iter', type=int, default=1000)
parser.add_argument('--n_steps', type=int, default=10000)
parser.add_argument('--n_epochs', type=int, default=15)
parser.add_argument('--model_save_dir', type=str, default='./best_model')

args = parser.parse_args()
assert ',' in args.ksize_list
args.ksize_list = [int(n) for n in args.ksize_list.split(',')]
args.emb_num = 50000
args.n_classes = 2

model = TextCNN(
    args.emb_num, args.emb_dim,
    ksize_list=args.ksize_list,
    n_filters_list=[args.n_filters] * len(args.ksize_list),
    n_classes=args.n_classes, dropout=args.dropout)


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config

@flow.global_function('train', get_train_config())
def train_job(text: tp.Numpy.Placeholder((args.batch_size, args.sequence_length), dtype=flow.int32),
              label: tp.Numpy.Placeholder((args.batch_size,), dtype=flow.int32)
              ) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = model.get_logits(text, is_train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(label, logits, name="softmax_loss")
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [args.lr])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return loss


@flow.global_function('predict', get_eval_config())
def eval_job(text: tp.Numpy.Placeholder((args.batch_size, args.sequence_length), dtype=flow.int32),
             label: tp.Numpy.Placeholder((args.batch_size,), dtype=flow.int32)
             ) -> Tuple[tp.Numpy, tp.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = model.get_logits(text, is_train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(label, logits, name="softmax_loss")

    return label, logits


def suffle_batch(data, label, batch_size):
    permu = np.random.permutation(len(data))
    data, label = data[permu], label[permu]

    batch_n = len(data) // batch_size

    x_batch = np.array([data[i * batch_size:i * batch_size + batch_size] for i in range(batch_n)], dtype=np.int32)
    y_batch = np.array([label[i * batch_size:i * batch_size + batch_size] for i in range(batch_n)], dtype=np.int32)

    return x_batch, y_batch


def acc(labels, logits, g):
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g["total"] += labels.shape[0]
    g["correct"] += right_count


def train(checkpoint):
    (train_data, train_labels), (test_data, test_labels) = load_imdb_data()

    with open('./imdb_word_index/imdb_word_index.json') as f:
        word_index = json.load(f)
    word_index = {k: (v + 2) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<UNK>"] = 1

    train_data = pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=args.sequence_length)
    test_data = pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=args.sequence_length)

    best_accuracy = 0.0
    best_epoch = 0
    for epoch in range(1, args.n_epochs + 1):
        print("[Epoch:{}]".format(epoch))
        data, label = suffle_batch(train_data, train_labels, args.batch_size)
        for i, (texts, labels) in enumerate(zip(data, label)):
            loss = train_job(texts, labels).mean()
            if i % 20 == 0:
                print(loss)

        data, label = suffle_batch(test_data, test_labels, args.batch_size)
        g = {"correct": 0, "total": 0}
        for i, (images, labels) in enumerate(zip(data, label)):
            labels, logits = eval_job(images, labels)
            acc(labels, logits, g)
        
        accuracy = g["correct"] * 100 / g["total"]
        print("[Epoch:{0:d} ] accuracy: {1:.1f}%".format(epoch, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            if not os.path.exists(args.model_save_dir):
                os.mkdir(args.model_save_dir)
            else:
                shutil.rmtree(args.model_save_dir)
                assert not os.path.exists(args.model_save_dir)
                os.mkdir(args.model_save_dir)
            print("Epoch:{} save best model.".format(best_epoch))
            checkpoint.save(args.model_save_dir)
    
    print("Epoch:{} get best accuracy:{}".format(best_epoch, best_accuracy))

if __name__ == '__main__':
    checkpoint = flow.train.CheckPoint()
    checkpoint.init()
    train(checkpoint)
