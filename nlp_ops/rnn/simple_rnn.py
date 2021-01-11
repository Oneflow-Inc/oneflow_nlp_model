import oneflow as flow

from .recurrent import rnn


def _FullyConnected(input_blob, weight_blob, bias_blob):
    output_blob = flow.matmul(input_blob, weight_blob)
    if bias_blob:
        output_blob = flow.nn.bias_add(output_blob, bias_blob)
    return output_blob


class SimpleRNNCell:

    def __init__(self, units,
                 activation=flow.math.tanh,
                 use_bias=True,
                 kernel_initializer=flow.glorot_uniform_initializer(),
                 recurrent_initializer=flow.glorot_normal_initializer(),  # should be orthogonal_initializer
                 bias_initializer=flow.zeros_initializer(),
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 dtype=flow.float32,
                 trainable=True,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.trainable = trainable
        self.dtype = dtype
        self.layer_index = kwargs['layer_index'] if 'layer_index' in kwargs else ''
        self.direction = kwargs['direction'] if 'layer_index' in kwargs else 'forward'

    def _build(self, inputs):
        input_size = inputs.shape[-1]
        units = self.units
        dtype = self.dtype
        trainable = self.trainable
        with flow.scope.namespace('layer' + str(self.layer_index)):
            with flow.scope.namespace(self.direction):
                self.kernel_blob = flow.get_variable(
                    name='input' + '-kernel',
                    shape=[input_size, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.kernel_regularizer,
                    initializer=self.kernel_initializer
                )

                self.recurrent_kernel_blob = flow.get_variable(
                    name='input' + '-recurrent-kernel',
                    shape=[units, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.recurrent_regularizer,
                    initializer=self.recurrent_initializer
                )

                self.bias_blob = flow.get_variable(
                    name='input' + '-bias',
                    shape=[units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.bias_regularizer,
                    initializer=self.bias_initializer
                ) if self.use_bias else None

    def __call__(self, inputs, states):
        self._build(inputs)

        hx = states[0]

        if 0 < self.dropout < 1.:
            inputs = flow.nn.dropout(inputs, rate=self.dropout)

        if 0 < self.recurrent_dropout < 1.:
            hx = flow.nn.dropout(hx, rate=self.recurrent_dropout)

        hy = _FullyConnected(inputs, self.kernel_blob, self.bias_blob)
        output = hy + _FullyConnected(hx, self.recurrent_kernel_blob, None)
        output = self.activation(output)

        return output, [output]


def simple_rnn(inputs,
               units,
               activation=flow.math.tanh,
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
               return_sequences=False,
               initial_state=None,
               **kwargs):
    return rnn(
        inputs,
        SimpleRNNCell(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        ),
        return_sequences=return_sequences, initial_state=initial_state, kwargs=kwargs
    )
