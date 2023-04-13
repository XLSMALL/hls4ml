import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Flatten, GRU, Activation, Embedding, Bidirectional, Flatten, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import Model
import hls4ml
import numpy as np
import os
os.environ['PATH'] = '/opt/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']

np.random.seed(3)
input_shape = (73, 70)  # Shape of the 'inputs' tensor 
# input_shape = (5, 2) # for testing
batch_size = 136  # Batch size
inputs_data = np.random.rand(batch_size, *input_shape)


initializer = tf.keras.initializers.VarianceScaling(distribution='normal')
regularizer = tf.keras.regularizers.L2(l=1)
inputs2model = inputs_data # 136, 73, 70
inputDim = 136
inputs2decoder = tf.stack([tf.zeros_like(inputs2model)[:, :, -1]
            for i in range(64)], axis=-1)
# inputs2decoder = np.zeros((17,73,64))
def create_model(inputs2model, inputs2decoder, initializer, regularizer):
  
  inputLayer =  Input(shape=inputs2model[0].shape)
  x = Dropout(0.05, name = 'initial_dropout')(inputLayer)
  x = Bidirectional(GRU(64, time_major=False, 
                        return_state=True, kernel_regularizer=regularizer, 
                        kernel_initializer=initializer), 
                    backward_layer=GRU(64, time_major=False, 
                                       return_state=True, go_backwards=True, 
                                       kernel_regularizer=regularizer, kernel_initializer=initializer), 
                    merge_mode='concat', name = 'Encoder_BidirectionalGRU')(x)[0]
  x = Dropout(0.05, name = 'postencoder_dropout')(x)
  x = Dense(64, kernel_regularizer=regularizer, kernel_initializer=initializer, name='dense_mean')(x)
  input_decoder = Input(shape=inputs2decoder[0].shape)
  x = GRU(64, return_sequences=True, 
          time_major=False, kernel_initializer=initializer, 
          kernel_regularizer=regularizer, name='decoder_GRU')(input_decoder, initial_state = x)
  x = Dropout(0.05, name = 'postdecoder_dropout')(x)
  z = Dense(4, use_bias = False, kernel_regularizer=regularizer, kernel_initializer=initializer, name='dense')(x)
  log_f = Dense(70, kernel_regularizer=regularizer, kernel_initializer=initializer, name='nerual_dense')(z)
  # z = Flatten()(z)
  # temp_1 = Dense(5,name='dense_temp1')(z)
  # log_f = Flatten()(log_f)
  # temp_2 = Dense(5,name='dense_temp2')(log_f)
  out = Concatenate()([z,log_f])

 # return Model(inputs = [inputLayer,input_decoder], outputs =[z, log_f])
  return Model(inputs = [inputLayer,input_decoder], outputs =out)

LFADs_keras = create_model(inputs2model,inputs2decoder, initializer, regularizer)
LFADs_keras.summary()
LFADs_keras.compile()

# LFADs = tf.keras.models.load_model("/home/xiaohan/HLS4ML_side_branch/hls4ml/test_LFADs/3_3_flatten.h5")
# LFADs.summary()
config = hls4ml.utils.config_from_keras_model(LFADs_keras, granularity='model', default_precision='ap_fixed<32,16>')
print("-----------------------------------")
print("Configuration")
'''plotting.print_dict(config)'''
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(LFADs_keras,
                                                       hls_config=config,
                                                       output_dir='/home/xiaohan/HLS4ML_side_branch/hls4ml/test_LFADs',
                                                       part='xc7v2000tflg1925-2')
print("done")
hls_model.compile()
# hls_out = hls_model.predict(x_in)
# print(hls_out)
#hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
# hls_model.build(csim=False)
