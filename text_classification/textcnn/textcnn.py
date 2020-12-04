from typing import Tuple
import oneflow as flow
import oneflow.typing as tp
import numpy as np

class TextCNN:
    def __init__(self, emb_sz, emb_dim, ksize_list, n_filters_list, n_classes, dropout):
        self.initializer = flow.random_normal_initializer(stddev=0.1)
        self.emb_sz = emb_sz
        self.emb_dim = emb_dim
        self.ksize_list= ksize_list
        self.n_filters_list = n_filters_list
        self.n_classes = n_classes
        self.dropout = dropout
        self.total_n_filters = sum(self.n_filters_list)

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
        data = flow.transpose(data, [0, 2, 1]) # BLH -> BHL
        data = flow.reshape(data, list(data.shape) + [1])
        seq_length = data.shape[2]
        pooled_list = []
        for i in range(len(self.n_filters_list)):
            ksz = self.ksize_list[i]
            n_filters = self.n_filters_list[i]
            conv = flow.layers.conv2d(data, n_filters, [ksz, 1], data_format="NCHW",
                                      kernel_initializer=self.initializer, name='conv-{}'.format(i)) #NCHW
            #conv = flow.layers.layer_norm(conv, name='ln-{}'.format(i))
            conv = flow.nn.relu(conv)
            pooled = flow.nn.max_pool2d(conv, [seq_length-ksz+1, 1], strides=1, padding='VALID', data_format="NCHW")
            pooled_list.append(pooled)
        pooled = flow.concat(pooled_list, 3)
        pooled = flow.reshape(pooled, [-1, self.total_n_filters])

        if is_train:
            pooled = flow.nn.dropout(pooled, rate=self.dropout)
        
        pooled = flow.layers.dense(pooled, self.total_n_filters, use_bias=True,
                                   kernel_initializer=self.initializer, name='dense-1')
        pooled = flow.nn.relu(pooled)
        logits = flow.layers.dense(pooled, self.n_classes, use_bias=True,
                                   kernel_initializer=self.initializer, name='dense-2')
        return logits

if __name__ == '__main__':
    @flow.global_function('predict', flow.function_config())
    def test_model(texts: tp.Numpy.Placeholder((32, 28), dtype=flow.int32),
                   labels: tp.Numpy.Placeholder((32, ), dtype=flow.int32)
        ) -> Tuple[tp.Numpy, tp.Numpy]:
        with flow.scope.placement("cpu", "0:0"):
            model = TextCNN(3000, 100, ksize_list=[2,3,5], n_filters_list=[100]*3, n_classes=2, dropout=0.1)
            logits = model.get_logits(texts, is_train=False)
            loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return logits, loss

    texts = np.random.randint(0, 3000, size=(32, 28)).astype(np.int32)
    labels = np.random.randint(0, 2, size=32, dtype=np.int32)
    checkpoint = flow.train.CheckPoint()
    checkpoint.init()
    logits, loss = test_model(texts, labels)
    print(logits.shape, loss.shape)
    print(type(logits), type(loss))
