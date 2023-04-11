import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Flatten, GRU, Activation, Embedding, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import Model
import numpy as np
import hls4ml
import os
os.environ['PATH'] = '/opt/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']



#####################################test 2 output with dense#####################################
###############################################################################################################
###############################################################################################################
np.random.seed(3)
input_shape = (5, 2)  # Shape of the 'inputs' tensor 
# input_shape = (5, 2) # for testing
batch_size = 1  # Batch size
inputs_data = np.random.rand(batch_size, *input_shape)

input_dense_shape = (2,)  # Shape of the 'input_dense' tensor
input_dense_data = np.random.rand(batch_size, *input_dense_shape)
# input_dense_data = np.zeros((batch_size, *input_dense_shape))


inputs = Input(shape=input_shape)
input_dense = Input(shape=input_dense_shape)
dense = Dense(1)(input_dense)
# gru_layer = GRU(units=1, return_sequences=True)(inputs, dense)
dense_1 = Dense(2)(inputs)
dense_2 = Dense(1)(dense)
dense_3 = Dense(1)(dense_1)
model_dense = tf.keras.Model(inputs=[input_dense,inputs], outputs=[dense_2, dense_3])


model_dense.summary()
model_dense.compile()
y_out = model_dense.predict([input_dense_data,inputs_data])
print("y_out prediction by keras is:")
print(y_out)




config = hls4ml.utils.config_from_keras_model(model_dense, granularity='model', default_precision='ap_fixed<64,32>')
print("-----------------------------------")
print("Configuration")
'''plotting.print_dict(config)'''
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(model_dense,
                                                       hls_config=config,
                                                       output_dir='/home/xiaohan/HLS4ML_side_branch/hls4ml/test_2_output',
                                                       #io_type = 'io_stream',
                                                       part='xc7z020clg400-1')
print("done")
hls_model.compile()
# hls_out = hls_model.predict(inputs_data)
# hls_out = hls_model.predict([input_dense_data,inputs_data])
# print(hls_out)







