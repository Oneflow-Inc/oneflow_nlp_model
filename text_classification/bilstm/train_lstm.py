import sys
import os
import argparse
import time
import json
from typing import Tuple

import numpy as np
import oneflow as flow
import oneflow.typing as tp

sys.path.append("../..")
from nlp_ops import bilstm
from text_classification.utils import pad_sequences, load_imdb_data

time_map = {}


def _colored_string(string: str, color: str or int) -> str:
    """在终端中显示一串有颜色的文字 [This code is copied from fitlog]

    :param string: 在终端中显示的文字
    :param color: 文字的颜色
    :return:
    """
    if isinstance(color, str):
        color = {
            "black": 30,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "purple": 35,
            "cyan": 36,
            "white": 37
        }[color]
        return "\033[%dm%s\033[0m" % (color, string)


class LSTMText:
    def __init__(self, emb_sz, emb_dim, hidden_size, nfc, n_classes):
        self.initializer = flow.random_normal_initializer(stddev=0.1)
        self.emb_sz = emb_sz
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.nfc = nfc

    def get_logits(self, inputs, is_train):
        emb_weight = flow.get_variable(
            'embedding-weight',
            shape=(self.emb_sz, self.emb_dim),
            dtype=flow.float32,
            trainable=is_train,
            reuse=False,
            initializer=self.initializer,
        )
        data = flow.gather(emb_weight, inputs, axis=0)
        data = bilstm(data, self.hidden_size)
        data = flow.layers.dense(data, self.nfc, use_bias=True,
                                 kernel_initializer=self.initializer, name='dense-1')
        logits = flow.layers.dense(data, self.n_classes, use_bias=True,
                                   kernel_initializer=self.initializer, name='dense-2')
        return logits


parser = argparse.ArgumentParser()
parser.add_argument('--emb_dim', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--nfc', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--sequence_length', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_load_dir', type=str, default='')
parser.add_argument('--model_save_every_n_iter', type=int, default=1000)
parser.add_argument('--n_steps', type=int, default=10000)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--model_save_dir', type=str, default='./save')

args = parser.parse_args()
args.emb_num = 50000
args.n_classes = 2

model = LSTMText(args.emb_num, args.emb_dim, hidden_size=args.hidden_size, nfc=args.nfc, n_classes=args.n_classes)


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
        logits = flow.nn.softmax(logits)
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


def train():
    print(_colored_string('Start Loading Data', 'green'))

    path = '../imdb'
    (train_data, train_labels), (test_data, test_labels) = load_imdb_data(path)

    with open(os.path.join(path, 'word_index.json')) as f:
        word_index = json.load(f)
    word_index = {k: (v + 2) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<UNK>"] = 1

    train_data = pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=args.sequence_length)
    test_data = pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=args.sequence_length)

    time_map['t2'] = time.time()
    print(_colored_string('Data Loading Time: %.2fs' % (time_map['t2'] - time_map['t1']), 'blue'))
    print(_colored_string('Start Training', 'green'))

    for epoch in range(1, args.n_epochs + 1):
        print("[Epoch:{}]".format(epoch))
        data, label = suffle_batch(train_data, train_labels, args.batch_size)
        for i, (texts, labels) in enumerate(zip(data, label)):
            loss = train_job(texts, labels).mean()
            # if i % 20 == 0:
            #     print(loss)

        data, label = suffle_batch(test_data, test_labels, args.batch_size)
        g = {"correct": 0, "total": 0}
        for i, (texts, labels) in enumerate(zip(data, label)):
            labels, logits = eval_job(texts, labels)
            acc(labels, logits, g)
        print("[Epoch:{0:d} ] accuracy: {1:.1f}%".format(epoch, g["correct"] * 100 / g["total"]))

    time_map['t3'] = time.time()
    print(_colored_string('Training Time: %.0fs' % (time_map['t3'] - time_map['t2']), 'blue'))


if __name__ == '__main__':
    time_map['t0'] = time.time()
    print(_colored_string('Start Compiling', 'green'))
    checkpoint = flow.train.CheckPoint()
    checkpoint.init()
    time_map['t1'] = time.time()
    print(_colored_string('Compiling Time: %.2fs' % (time_map['t1'] - time_map['t0']), 'blue'))
    train()
