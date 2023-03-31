import numpy as np

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

from hls4ml.model.types import Quantizer
from hls4ml.model.types import IntegerPrecisionType


@keras_handler('Bidirectional')
def parse_bidirectional_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'Bidirectional')

    layer = parse_default_keras_layer(keras_layer, input_names)
    # print(keras_layer)
    # print("check", keras_layer['config'])
    layer['layer_type'] = keras_layer['config']['layer']['config']['name'] # get forward & backward layers type
    #layer['backward_layer'] = keras_layer['config']['backward_layer']
    layer['return_state'] = keras_layer['config']['layer']['config']['return_state']
    layer['merge_mode'] = keras_layer['config']['merge_mode']

    layer['n_out_forward'] = keras_layer['config']['layer']['config']['units']
    layer['n_out_backward'] = keras_layer['config']['backward_layer']['config']['units']
    
    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]
    layer['n_out'] = keras_layer['config']['backward_layer']['config']['units']
    layer['activation'] = keras_layer['config']['layer']['config']['activation']
    layer['recurrent_activation'] = keras_layer['config']['layer']['config']['recurrent_activation']
    layer['initial_state'] = 0
    # print("test test test")
    # if (len(keras_layer['inbound_nodes'][0]) > 1):
    #     layer['initial_state'] = 1
    # else:
    #     layer['initial_state'] = 0
    # forward_layer_unit = layer['layer']['config']['units']
    # backward_layer_unit = layer['backward_layer']['config']['units']

    #if layer['layer']['config']['return_state']:
    if layer['return_state']:
        # output_shape = [input_shapes[0][0], forward_layer_unit + backward_layer_unit]
        output_shape = [input_shapes[0][0], layer['n_out_forward'] + layer['n_out_backward']]
        # output_shape_1 = [input_shapes[0][0], forward_layer_unit]
        # output_shape_2 = [input_shapes[0][0], backward_layer_unit]
        # output_shape = [output_shape_0, output_shape_1, output_shape_2]
    else:
        # output_shape = [input_shapes[0][0], forward_layer_unit + backward_layer_unit]
        output_shape = [input_shapes[0][0], layer['n_out_forward'] + layer['n_out_backward']]
    return layer, output_shape
