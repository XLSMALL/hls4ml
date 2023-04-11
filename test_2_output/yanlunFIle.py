from tensorflow.keras.layers import Conv2D, Concatenate, Input, Reshape, Flatten
from tensorflow.keras.models import Model

inp1 = Input((8,8,3))
inp2 = Input((10,10,5))
x1 = Conv2D(6,(3,3),padding='same')(inp1) 
x2 = Conv2D(4,(3,3),padding='same')(inp2) 
x1 = Flatten()(x1)
x2 = Flatten()(x2)
model = Model(inputs=[inp1, inp2], outputs=[x1,x2])
model.summary()

from tensorflow.keras import Model
#deleted_model = Model(model.input, model.get_layer('conv2d_3').output)
import hls4ml
#import plotting
import pprint
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation', 'Input'])
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(rounding_mode='AP_RND_CONV')
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(saturation_mode='AP_SAT')

config = hls4ml.utils.config_from_keras_model(model, granularity='Model')
#config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
#config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
config['Model']['Strategy'] = 'resource'
config['Model']['Precision'] = 'ap_fixed<32,16>'
#pprint.pprint(config)
#config['LayerName']['conv2d_batchnorm_8']['Precision']['weight'] = 'ap_fixed<16,4>'
#config['LayerName']['conv2d_batchnorm_8']['Precision']['bias'] = 'ap_fixed<16,4>'
print("-----------------------------------")
#plotting.print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir='/home/xiaohan/HLS4ML_side_branch/hls4ml/test_2_output',
                                                       io_type='io_parallel',
                                                       part='xcu250-figd2104-2L-e')
                                                       #part='xczu9eg-ffvb1156-2-e')
                                                       #backend='VivadoAccelerator',
                                                       #board='pynq-z2')
hls_model.compile()