import oneflow as flow

from .recurrent import rnn


def _FullyConnected(input_blob, weight_blob, bias_blob):
    output_blob = flow.matmul(input_blob, weight_blob)
    if bias_blob:
        output_blob = flow.nn.bias_add(output_blob, bias_blob)
    return output_blob


class LSTMCell:
    
    def __init__(self, units,
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
        with flow.scope.namespace('layer' + str(self.layer_index)):
            with flow.scope.namespace(self.direction):
                self.kernel_blob_i = flow.get_variable(
                    name='input' + '-kernel',
                    shape=[input_size, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.kernel_regularizer,
                    initializer=self.kernel_initializer
                )
                
                self.recurrent_kernel_blob_i = flow.get_variable(
                    name='input' + '-recurrent-kernel',
                    shape=[units, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.recurrent_regularizer,
                    initializer=self.recurrent_initializer
                )
                
                self.bias_blob_i = flow.get_variable(
                    name='input' + '-bias',
                    shape=[units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.bias_regularizer,
                    initializer=flow.zeros_initializer() if self.unit_forget_bias else self.bias_initializer
                ) if self.use_bias else None
                
                self.kernel_blob_f = flow.get_variable(
                    name='forget' + '-kernel',
                    shape=[input_size, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.kernel_regularizer,
                    initializer=self.kernel_initializer
                )
                
                self.recurrent_kernel_blob_f = flow.get_variable(
                    name='forget' + '-recurrent-kernel',
                    shape=[units, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.recurrent_regularizer,
                    initializer=self.recurrent_initializer
                )
                
                self.bias_blob_f = flow.get_variable(
                    name='forget' + '-bias',
                    shape=[units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.bias_regularizer,
                    initializer=flow.ones_initializer() if self.unit_forget_bias else self.bias_initializer
                ) if self.use_bias else None
                
                self.kernel_blob_c = flow.get_variable(
                    name='cell' + '-kernel',
                    shape=[input_size, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.kernel_regularizer,
                    initializer=self.kernel_initializer
                )
                
                self.recurrent_kernel_blob_c = flow.get_variable(
                    name='cell' + '-recurrent-kernel',
                    shape=[units, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.recurrent_regularizer,
                    initializer=self.recurrent_initializer
                )
                
                self.bias_blob_c = flow.get_variable(
                    name='cell' + '-bias',
                    shape=[units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.bias_regularizer,
                    initializer=flow.zeros_initializer() if self.unit_forget_bias else self.bias_initializer
                ) if self.use_bias else None
                
                self.kernel_blob_o = flow.get_variable(
                    name='output' + '-kernel',
                    shape=[input_size, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.kernel_regularizer,
                    initializer=self.kernel_initializer
                )
                
                self.recurrent_kernel_blob_o = flow.get_variable(
                    name='output' + '-recurrent-kernel',
                    shape=[units, units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.recurrent_regularizer,
                    initializer=self.recurrent_initializer
                )
                
                self.bias_blob_o = flow.get_variable(
                    name='output' + '-bias',
                    shape=[units],
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=self.bias_regularizer,
                    initializer=flow.zeros_initializer() if self.unit_forget_bias else self.bias_initializer
                ) if self.use_bias else None
    
    def __call__(self, inputs, states):
        self._build(inputs)
        
        hx = states[0]
        cx = states[1]
        
        if 0 < self.dropout < 1.:
            inputs_i = flow.nn.dropout(inputs, rate=self.dropout)
            inputs_f = flow.nn.dropout(inputs, rate=self.dropout)
            inputs_c = flow.nn.dropout(inputs, rate=self.dropout)
            inputs_o = flow.nn.dropout(inputs, rate=self.dropout)
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
        
        if 0 < self.recurrent_dropout < 1.:
            hx_i = flow.nn.dropout(hx, rate=self.recurrent_dropout)
            hx_f = flow.nn.dropout(hx, rate=self.recurrent_dropout)
            hx_c = flow.nn.dropout(hx, rate=self.recurrent_dropout)
            hx_o = flow.nn.dropout(hx, rate=self.recurrent_dropout)
        else:
            hx_i = hx
            hx_f = hx
            hx_c = hx
            hx_o = hx
        
        x_i = _FullyConnected(inputs_i, self.kernel_blob_i, self.bias_blob_i)  # input gate
        x_f = _FullyConnected(inputs_f, self.kernel_blob_f, self.bias_blob_f)  # forget gate
        x_c = _FullyConnected(inputs_c, self.kernel_blob_c, self.bias_blob_c)  # cell state
        x_o = _FullyConnected(inputs_o, self.kernel_blob_o, self.bias_blob_o)  # output gate
        
        h_i = _FullyConnected(hx_i, self.recurrent_kernel_blob_i, None)
        h_f = _FullyConnected(hx_f, self.recurrent_kernel_blob_f, None)
        h_c = _FullyConnected(hx_c, self.recurrent_kernel_blob_c, None)
        h_o = _FullyConnected(hx_o, self.recurrent_kernel_blob_o, None)
        
        x_i = x_i + h_i
        x_f = x_f + h_f
        x_c = x_c + h_c
        x_o = x_o + h_o
        
        x_i = self.recurrent_activation(x_i)
        x_f = self.recurrent_activation(x_f)
        cell_gate = self.activation(x_c)
        x_o = self.recurrent_activation(x_o)
        
        cy = x_f * cx + x_i * cell_gate
        
        hy = x_o * self.activation(cy)
        
        return hy, (hy, cy)


def lstm(inputs,
         units,
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
         return_sequences=False,
         initial_state=None,
         **kwargs):
    return rnn(inputs,
               LSTMCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
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
               return_sequences=return_sequences, initial_state=initial_state, kwargs=kwargs)


def bilstm(inputs,
           units,
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
           return_sequences=False,
           initial_state=None,
           **kwargs):
    forward = rnn(inputs,
                  LSTMCell(units,
                           activation=activation,
                           recurrent_activation=recurrent_activation,
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
                  return_sequences=return_sequences, initial_state=initial_state, kwargs=kwargs)
    
    backward = rnn(inputs,
                   LSTMCell(units,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
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
                   return_sequences=return_sequences, initial_state=initial_state, kwargs=kwargs)
    
    backward = flow.reverse(backward, axis=1)
    
    outputs = forward + backward
    
    return outputs
