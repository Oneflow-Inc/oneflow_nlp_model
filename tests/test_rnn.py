import sys
import oneflow as flow
import oneflow.typing as tp
import numpy as np
import unittest

import test_global_storage

sys.path.append('..')

from nlp_ops.rnn import rnn, SimpleRNNCell, LSTMCell, simple_rnn, lstm, bilstm


def watched_simple_rnn(inputs, units, return_sequences=False, initial_state=None):
    simple_rnn_cell = SimpleRNNCell(units)

    outputs = rnn(inputs, simple_rnn_cell, return_sequences=return_sequences, initial_state=initial_state)

    flow.watch(simple_rnn_cell.kernel_blob, test_global_storage.Setter("kernel_blob"))
    flow.watch(simple_rnn_cell.recurrent_kernel_blob, test_global_storage.Setter("recurrent_kernel_blob"))
    flow.watch(simple_rnn_cell.bias_blob, test_global_storage.Setter("bias_blob"))

    return outputs


def watched_lstm(inputs, units, return_sequences=False, initial_state=None):
    lstm_cell = LSTMCell(units)

    outputs = rnn(inputs, lstm_cell, return_sequences=return_sequences, initial_state=initial_state)

    flow.watch(lstm_cell.kernel_blob_i, test_global_storage.Setter("kernel_blob_i"))
    flow.watch(lstm_cell.kernel_blob_f, test_global_storage.Setter("kernel_blob_f"))
    flow.watch(lstm_cell.kernel_blob_c, test_global_storage.Setter("kernel_blob_c"))
    flow.watch(lstm_cell.kernel_blob_o, test_global_storage.Setter("kernel_blob_o"))

    flow.watch(lstm_cell.recurrent_kernel_blob_i, test_global_storage.Setter("recurrent_kernel_blob_i"))
    flow.watch(lstm_cell.recurrent_kernel_blob_f, test_global_storage.Setter("recurrent_kernel_blob_f"))
    flow.watch(lstm_cell.recurrent_kernel_blob_c, test_global_storage.Setter("recurrent_kernel_blob_c"))
    flow.watch(lstm_cell.recurrent_kernel_blob_o, test_global_storage.Setter("recurrent_kernel_blob_o"))

    flow.watch(lstm_cell.bias_blob_i, test_global_storage.Setter("bias_blob_i"))
    flow.watch(lstm_cell.bias_blob_f, test_global_storage.Setter("bias_blob_f"))
    flow.watch(lstm_cell.bias_blob_c, test_global_storage.Setter("bias_blob_c"))
    flow.watch(lstm_cell.bias_blob_o, test_global_storage.Setter("bias_blob_o"))

    return outputs


flow.config.gpu_device_num(1)
flow.config.enable_debug_mode(True)


def get_test_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


def oneflow_session(func):
    def test_func(*args, **kwargs):
        func(*args, **kwargs)
        flow.clear_default_session()

    return test_func


class TestAllRNN(unittest.TestCase):

    @oneflow_session
    def test_simple_rnn_cell(self):
        @flow.global_function('predict', get_test_config())
        def inference_watched_simple_rnn(sentence: tp.Numpy.Placeholder(shape=(32, 18, 64), dtype=flow.float32)
                                         ) -> tp.Numpy:
            output = watched_simple_rnn(sentence, 15, return_sequences=True)

            return output

        sentence_in = np.random.uniform(-10, 10, (32, 18, 64)).astype(np.float32)
        output_of = inference_watched_simple_rnn(sentence_in)

        from tensorflow.keras import layers
        from tensorflow import keras

        inputs = keras.Input(shape=(18, 64))
        x = layers.SimpleRNN(15, return_sequences=True, name="simple_rnn_one")(inputs)

        kernel_blob = test_global_storage.Get("kernel_blob")
        recurrent_kernel_blob = test_global_storage.Get("recurrent_kernel_blob")
        bias_blob = test_global_storage.Get("bias_blob")

        model = keras.Model(inputs, x)
        model.get_layer("simple_rnn_one").set_weights([kernel_blob, recurrent_kernel_blob, bias_blob])
        output_tf = model.predict(sentence_in)

        assert (np.allclose(output_of, output_tf, rtol=1e-04, atol=1e-04))

    @oneflow_session
    def test_lstm_cell(self):
        @flow.global_function('predict', get_test_config())
        def inference_watched_lstm(sentence: tp.Numpy.Placeholder(shape=(32, 18, 64), dtype=flow.float32)
                                   ) -> tp.Numpy:
            output = watched_lstm(sentence, 15, return_sequences=True)

            return output

        sentence_in = np.random.uniform(-10, 10, (32, 18, 64)).astype(np.float32)
        output_of = inference_watched_lstm(sentence_in)

        from tensorflow.keras import layers
        from tensorflow import keras

        inputs = keras.Input(shape=(18, 64))
        x = layers.LSTM(15, return_sequences=True, recurrent_activation='sigmoid', name="lstm_one")(inputs)

        kernel_blob_i = test_global_storage.Get("kernel_blob_i")
        kernel_blob_f = test_global_storage.Get("kernel_blob_f")
        kernel_blob_c = test_global_storage.Get("kernel_blob_c")
        kernel_blob_o = test_global_storage.Get("kernel_blob_o")

        kernel = np.concatenate((kernel_blob_i, kernel_blob_f, kernel_blob_c, kernel_blob_o), axis=1)

        recurrent_kernel_blob_i = test_global_storage.Get("recurrent_kernel_blob_i")
        recurrent_kernel_blob_f = test_global_storage.Get("recurrent_kernel_blob_f")
        recurrent_kernel_blob_c = test_global_storage.Get("recurrent_kernel_blob_c")
        recurrent_kernel_blob_o = test_global_storage.Get("recurrent_kernel_blob_o")

        recurrent_kernel = np.concatenate(
            (recurrent_kernel_blob_i, recurrent_kernel_blob_f, recurrent_kernel_blob_c, recurrent_kernel_blob_o),
            axis=1)

        bias_blob_i = test_global_storage.Get("bias_blob_i")
        bias_blob_f = test_global_storage.Get("bias_blob_f")
        bias_blob_c = test_global_storage.Get("bias_blob_c")
        bias_blob_o = test_global_storage.Get("bias_blob_o")

        bias_1 = np.concatenate((bias_blob_i, bias_blob_f, bias_blob_c, bias_blob_o))

        model = keras.Model(inputs, x)
        model.get_layer("lstm_one").set_weights([kernel, recurrent_kernel, bias_1])
        output_tf = model.predict(sentence_in)

        assert (np.allclose(output_of, output_tf, rtol=1e-04, atol=1e-04))

    @oneflow_session
    def test_output_size(self):
        @flow.global_function('predict', get_test_config())
        def inference_simple_rnn(sentence: tp.Numpy.Placeholder((32, 18, 64), dtype=flow.float32)
                                 ) -> tp.Numpy:
            output = simple_rnn(sentence, 15, return_sequences=True)

            return output

        @flow.global_function('predict', get_test_config())
        def inference_lstm(sentence: tp.Numpy.Placeholder((32, 18, 64), dtype=flow.float32)
                           ) -> tp.Numpy:
            output = lstm(sentence, 15, return_sequences=True)

            return output

        @flow.global_function('predict', get_test_config())
        def inference_bilstm(sentence: tp.Numpy.Placeholder((32, 18, 64), dtype=flow.float32)
                             ) -> tp.Numpy:
            output = bilstm(sentence, 15, return_sequences=True)

            return output

        sentence_in = np.random.uniform(-10, 10, (32, 18, 64)).astype(np.float32)
        output_0 = inference_simple_rnn(sentence_in)
        output_1 = inference_lstm(sentence_in)
        output_2 = inference_bilstm(sentence_in)

        self.assertEqual(output_0.shape, output_1.shape, output_2.shape)
