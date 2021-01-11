import oneflow as flow


class RNN:

    def __init__(self, cell, return_sequences=False, return_state=False, **kwargs):
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        if "initial_state" in kwargs:
            self.initial_state = kwargs["initial_state"]
        else:
            self.initial_state = None

    def __call__(self, inputs):

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        input_size = inputs.shape[2]

        if self.initial_state:
            states = self.initial_state
        else:
            states = [flow.constant(0, dtype=flow.float32, shape=[batch_size, self.cell.units]),
                      flow.constant(0, dtype=flow.float32, shape=[batch_size, self.cell.units])]

        successive_outputs = []
        successive_states = []

        for index in range(seq_len):
            input_now = flow.slice(inputs, [None, index, 0], [None, 1, input_size])
            input_now = flow.reshape(input_now, [-1, input_size])
            output, states = self.cell(input_now, states)

            output = flow.reshape(output, [-1, 1, self.cell.units])
            successive_outputs.append(output)
            successive_states.append(states)
        last_output = successive_outputs[-1]
        new_states = successive_states[-1]  # 可能可以用于 stateful 方式
        outputs = flow.concat(successive_outputs, axis=1)

        if self.return_sequences:
            return outputs
        else:
            return flow.reshape(last_output, [-1, self.cell.units])


def rnn(inputs, cell, return_sequences=False, return_state=False, **kwargs):
    """
        与 OneFlow 的设计风格进行统一
    """
    return RNN(cell, return_sequences=return_sequences, return_state=return_state, **kwargs)(inputs)
