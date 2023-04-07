import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Flatten, GRU, Activation, Embedding, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import Model
import numpy as np
import hls4ml
import os
os.environ['PATH'] = '/opt/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']

# x_in = tf.random.uniform(shape=[5,10], seed = 17)
# x_in_dense = x_in
# x_in = x_in.numpy()
# x_in_dense = x_in_dense.numpy()
# x_in = x_in.reshape((1,5,10))
# inputs2gru =tf.stack([tf.zeros_like(x_in)[:, :, -1]
#             for i in range(2)], axis=-1)
# initial_state = np.ones((1,2))
# initial_state = tf.convert_to_tensor(initial_state, dtype=tf.float32)
# print(x_in[0][0].shape)



#####test model
# initializer = tf.keras.initializers.VarianceScaling(distribution='normal')
# regularizer = tf.keras.regularizers.L2(l=1)
# # inputs2model = x_in # 136, 73, 70
# # inputDim = 136
# # inputs2decoder = tf.stack([tf.zeros_like(inputs2model)[:, :, -1]
# #             for i in range(2)], axis=-1)
# # # inputs2decoder = np.zeros((17,73,64))
# def create_model(inputs2model, inputs2decoder, initializer, regularizer):
  
#   inputLayer =  Input(shape=inputs2model[0].shape)
#   x = Bidirectional(GRU(2, time_major=False, 
#                         return_state=True, kernel_regularizer=regularizer, 
#                         kernel_initializer=initializer), 
#                     backward_layer=GRU(2, time_major=False, 
#                                        return_state=True, go_backwards=True, 
#                                        kernel_regularizer=regularizer, kernel_initializer=initializer), 
#                     merge_mode='concat', name = 'Encoder_BidirectionalGRU')(inputLayer)[0]
#   x = Dense(2, kernel_regularizer=regularizer, kernel_initializer=initializer, name='dense_mean')(x)
#   input_decoder = Input(shape=inputs2decoder[0].shape)
#   x = GRU(2, return_sequences=True, 
#           time_major=False, kernel_initializer=initializer, 
#           kernel_regularizer=regularizer, name='decoder_GRU')(input_decoder, initial_state = x)
# #   z = Dense(4, use_bias = False, kernel_regularizer=regularizer, kernel_initializer=initializer, name='dense')(x)
# #   return Model(inputs = [inputLayer,input_decoder], outputs =[x, z])
#   return Model(inputs = [inputLayer,input_decoder], outputs =[x])

# encoder = create_model(x_in,x_in, initializer, regularizer)

#####################################test gru intiial_state with dense#####################################
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
gru_layer = GRU(units=1, return_sequences=True)(inputs, dense)
encoder = tf.keras.Model(inputs=[input_dense,inputs], outputs=gru_layer)


fix_weight_model = tf.keras.models.load_model("/home/xiaohan/HLS4ML_side_branch/hls4ml/test_gru/fix_weight_model.h5")
# print("layer2w", fix_weight_model.layers[3])
encoder.layers[2].set_weights(fix_weight_model.layers[2].get_weights())
encoder.layers[3].set_weights(fix_weight_model.layers[3].get_weights())

#####################################test gru intiial_state only#####################################
###############################################################################################################
###############################################################################################################
# np.random.seed(2)
# input_shape = (5, 2)  # Shape of the 'inputs' tensor 
# # input_shape = (5, 2) # for testing
# batch_size = 1  # Batch size
# inputs_data = np.random.rand(batch_size, *input_shape)

# input_dense_shape = (1,)  # Shape of the 'input_dense' tensor
# input_dense_data = np.random.rand(batch_size, *input_dense_shape)


# inputs = Input(shape=input_shape)
# # input_dense = Input(shape=input_dense_shape)
# gru_layer = GRU(units=1, return_sequences=True)(inputs)
# # gru_layer = GRU(units=1, return_sequences=True)(inputs)
# encoder = tf.keras.Model(inputs=[inputs], outputs=gru_layer)
# # encoder = tf.keras.Model(inputs=inputs, outputs=gru_layer)
# # print(fix_weight_model.layers[1].get_weights())
# # fix_weight_model_gru_only.summary()
# # fix_weight_model_gru_only.compile()
# # saved_model_dir = "/home/xiaohan/ACME_hls4ml/hls4ml/test_GRU/fix_weight_model_gru_only.h5"
# # fix_weight_model_gru_only.save(saved_model_dir)

# fix_weight_model_gru_only = tf.keras.models.load_model("/home/xiaohan/ACME_hls4ml/hls4ml/test_GRU/fix_weight_model_gru_only.h5")
# # # print("layer2w", fix_weight_model.layers[3])
# encoder.layers[2].set_weights(fix_weight_model_gru_only.layers[2].get_weights())


encoder.summary()
encoder.compile()
y_out = encoder.predict([input_dense_data,inputs_data])
print("y_out prediction by keras is:")
print(y_out)

intermediate_layer_model = tf.keras.Model(inputs=input_dense,outputs=dense)
intermediate_layer_model.layers[1].set_weights(fix_weight_model.layers[2].get_weights())
intermediate_layer_model.compile()
dense_out = intermediate_layer_model.predict(input_dense_data)
print("dense_out prediction by keras is:")
print(dense_out)
# inputs = Input(shape=input_shape)
# input_dense = Input(shape=input_dense_shape)
# dense = Dense(1)(input_dense)
# gru_layer = GRU(units=1, return_sequences=True)(inputs)
# encoder = tf.keras.Model(inputs=[inputs], outputs=gru_layer)

# encoder.summary()
# encoder.compile()
# y_out = encoder.predict(inputs_data)
# print("y_out prediction by keras is:")
# print(y_out)

### new model for testing gru_stack
# inputs = Input(shape=input_shape)
# gru_layer = GRU(units=64, return_sequences=True)(inputs) # set return_sequences = True to use gru_stack in hls4ml
# # gru_layer = GRU(units=2, return_sequences=True)(inputs) # for testing
# encoder = tf.keras.Model(inputs=[inputs], outputs=gru_layer)



config = hls4ml.utils.config_from_keras_model(encoder, granularity='model', default_precision='ap_fixed<64,32>')
print("-----------------------------------")
print("Configuration")
'''plotting.print_dict(config)'''
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(encoder,
                                                       hls_config=config,
                                                       output_dir='/home/xiaohan/HLS4ML_side_branch/hls4ml/test_gru',
                                                       #io_type = 'io_stream',
                                                       part='xc7z020clg400-1')
print("done")
hls_model.compile()
# hls_out = hls_model.predict(inputs_data)
hls_out = hls_model.predict([input_dense_data,inputs_data])
print(hls_out)



# #hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
# hls_model.build(csim=False)
# print("----------------------------------------")
# print("----------------------------------------")
# print("----------------------------------------")
# print("----------------------------------------")
# print("----------------------------------------")
# print("----------------SYN-DONE----------------")
# print("----------------------------------------")
# print("----------------------------------------")
# print("----------------------------------------")
# print("----------------------------------------")
# print("----------------------------------------")
# hls4ml.report.read_vivado_report('/home/xiaohan/ACME_hls4ml/hls4ml/test_GRU')