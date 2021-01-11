"""
Cached LSTM operator

https://arxiv.org/abs/1610.04989

@article{xu2016cached,
  title={Cached long short-term memory neural networks for document-level sentiment classification},
  author={Xu, Jiacheng and Chen, Danlu and Qiu, Xipeng and Huang, Xuangjing},
  journal={arXiv preprint arXiv:1610.04989},
  year={2016}
}

"""

import oneflow as flow
from .recurrent import rnn


def _FullyConnected(input_blob, weight_blob, bias_blob):
    output_blob = flow.matmul(input_blob, weight_blob)
    if bias_blob:
        output_blob = flow.nn.bias_add(output_blob, bias_blob)
    return output_blob


class CachedLSTMCell:

    def __init__(self, group_num, units,
                 activation=flow.math.tanh,
                 recurrent_activation=flow.math.sigmoid,
                 use_bias=True,
                 kernel_initializer=flow.glorot_uniform_initializer(),
                 recurrent_initializer=flow.glorot_normal_initializer(),  # should be orthogonal_initializer
                 bias_initializer=flow.zeros_initializer(),
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 dtype=flow.float32,
                 trainable=True,
                 **kwargs):
        self.group_num = group_num
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.dropout = min(1., max(0., dropout))
        self.dtype = dtype
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.trainable = trainable
        self.layer_index = kwargs['layer_index'] if 'layer_index' in kwargs else ''
        self.direction = kwargs['direction'] if 'layer_index' in kwargs else 'forward'

    def _build(self, inputs):
        input_size = inputs.shape[-1]
        units = self.units
        dtype = self.dtype
        trainable = self.trainable
        self.W_r = []
        self.W_o = []
        self.W_c = []
        self.bias_W_r = []
        self.bias_W_o = []
        self.bias_W_c = []
        self.U_f = []
        self.U_o = []
        self.U_c = []
        for k in range(self.group_num):
            self.W_r.append(flow.get_variable(
                name='W_r-{}'.format(k),
                shape=[input_size, units],
                dtype=dtype,
                trainable=trainable,
                regularizer=self.kernel_regularizer,
                initializer=self.kernel_initializer
            ))
            self.W_o.append(flow.get_variable(
                name='W_o-{}'.format(k),
                shape=[input_size, units],
                dtype=dtype,
                trainable=trainable,
                regularizer=self.kernel_regularizer,
                initializer=self.kernel_initializer
            ))
            self.W_c.append(flow.get_variable(
                name='W_c-{}'.format(k),
                shape=[input_size, units],
                dtype=dtype,
                trainable=trainable,
                regularizer=self.kernel_regularizer,
                initializer=self.kernel_initializer
            ))
            self.bias_W_r.append(flow.get_variable(
                name='bias_W_r-{}'.format(k),
                shape=[units],
                dtype=dtype,
                trainable=trainable,
                regularizer=self.bias_regularizer,
                initializer=flow.zeros_initializer() if self.unit_forget_bias else self.bias_initializer
            ) if self.use_bias else None)
            self.bias_W_o.append(flow.get_variable(
                name='bias_W_o-{}'.format(k),
                shape=[units],
                dtype=dtype,
                trainable=trainable,
                regularizer=self.bias_regularizer,
                initializer=flow.zeros_initializer() if self.unit_forget_bias else self.bias_initializer
            ) if self.use_bias else None)
            self.bias_W_c.append(flow.get_variable(
                name='bias_W_c-{}'.format(k),
                shape=[units],
                dtype=dtype,
                trainable=trainable,
                regularizer=self.bias_regularizer,
                initializer=flow.zeros_initializer() if self.unit_forget_bias else self.bias_initializer
            ) if self.use_bias else None)

            self.U_f.append([])
            self.U_o.append([])
            self.U_c.append([])
            for j in range(self.group_num):
                self.U_f[-1].append(flow.get_variable(
                    name='U_f-{}-{}'.format(j, k),
                    shape=[units, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.recurrent_regularizer,
                    initializer=self.recurrent_initializer
                ))
                self.U_o[-1].append(flow.get_variable(
                    name='U_o-{}-{}'.format(j, k),
                    shape=[units, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.recurrent_regularizer,
                    initializer=self.recurrent_initializer
                ))
                self.U_c[-1].append(flow.get_variable(
                    name='U_c-{}-{}'.format(j, k),
                    shape=[units, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.recurrent_regularizer,
                    initializer=self.recurrent_initializer
                ))

    def __call__(self, inputs, states):
        self._build(inputs)

        hx = states[0]
        cx = states[1]

        hy = []
        cy = []

        if 0 < self.dropout < 1.:
            inputs_f = flow.nn.dropout(inputs, rate=self.dropout)
            inputs_o = flow.nn.dropout(inputs, rate=self.dropout)
            inputs_c = flow.nn.dropout(inputs, rate=self.dropout)
        else:
            inputs_f = inputs
            inputs_o = inputs
            inputs_c = inputs

        if 0 < self.recurrent_dropout < 1.:
            hx_f = [flow.nn.dropout(hx[k], rate=self.recurrent_dropout) for k in range(self.group_num)]
            hx_o = [flow.nn.dropout(hx[k], rate=self.recurrent_dropout) for k in range(self.group_num)]
            hx_c = [flow.nn.dropout(hx[k], rate=self.recurrent_dropout) for k in range(self.group_num)]
        else:
            hx_f = hx
            hx_o = hx
            hx_c = hx

        for k in range(self.group_num):
            x_f = _FullyConnected(inputs_f, self.W_r[k], self.bias_W_r[k])
            x_o = _FullyConnected(inputs_o, self.W_o[k], self.bias_W_o[k])
            x_c = _FullyConnected(inputs_c, self.W_c[k], self.bias_W_c[k])

            for j in range(self.group_num):
                x_f = x_f + _FullyConnected(hx_f[j], self.U_f[j][k], None)
                x_o = x_o + _FullyConnected(hx_o[j], self.U_o[j][k], None)
                x_c = x_c + _FullyConnected(hx_c[j], self.U_c[j][k], None)

            x_f = self.recurrent_activation(x_f)
            r = 1 / self.group_num * x_f + k / self.group_num  # (k-1)/group_num in paper
            o = self.recurrent_activation(x_o)
            c_tilde = self.activation(x_c)
            c = (1 - r) * cx[k] + r * c_tilde
            h = o * self.activation(c)
            hy.append(h)
            cy.append(c)

        return hy[0], (hy, cy)


def cached_lstm(inputs,
                group_num,
                units,
                dropout=0.,
                recurrent_dropout=0.,
                return_sequences=False,
                initial_state=None):
    return rnn(
        inputs,
        CachedLSTMCell(
            group_num,
            units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        ),
        return_sequences=return_sequences,
        initial_state=initial_state if initial_state is not None else [
            [flow.constant(0, dtype=flow.float32, shape=[inputs.shape[0], units]) for _ in range(group_num)],
            [flow.constant(0, dtype=flow.float32, shape=[inputs.shape[0], units]) for _ in range(group_num)]
        ]
    )
