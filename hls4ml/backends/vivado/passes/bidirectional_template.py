from hls4ml.backends.backend import get_backend
from hls4ml.model.layers import Bidirectional
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

birecr_mult_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t.name} accum_t;
    typedef {forward_gru_gru_cell_2_bias_t.name} bias_t;
    typedef {forward_gru_gru_cell_2_weight_t.name} weight_t;
    typedef ap_{index_t} index_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

activ_birecr_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef ap_{table_t} table_t;
}};\n"""

birecr_activ_config_template = """struct {type}_config{index}_recr : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef ap_{table_t} table_t;
}};\n"""
#     recr_config_template = """struct config{index} : nnet::{recr_type}_config {{
# LSTM + GRU templates
bidir_recr_config_template = """struct config{index}_{direction} : nnet::{recr_type}_config {{
    typedef {accum_t.name} accum_t;
    typedef {forward_gru_gru_cell_2_weight_t.name} weight_t;  // Matrix
    typedef {forward_gru_gru_cell_2_bias_t.name} bias_t;  // Vector
    typedef {config_mult_t1} mult_config1;
    typedef {config_mult_t2} mult_config2;
    typedef {recr_act_t} ACT_CONFIG_{RECR_TYPE};
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;
    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;
    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_state = {n_state};
    static const unsigned n_sequence = {n_sequence};
    static const unsigned n_sequence_out = {n_sequence_out};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned initial_state = {initial_state};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
}};\n"""

# bidir_config_template = """struct config{index} : nnet::bidirectional_config {{
#     typedef {accum_t.name} accum_t;
#     typedef {forward_gru_gru_cell_2_weight_t.name} weight_t;  // Matrix
#     typedef {forward_gru_gru_cell_2_bias_t.name} bias_t;  // Vector
        
#     typedef {config_frnn_layer} config_rnn_layer_f;
#     typedef {config_brnn_layer} config_rnn_layer_b;

    # typedef {config_frnn_layer} FLAYER_CONFIG_{RECR_TYPE};
    # template<class x_T, class y_T, class config_T>
    # using foward_layer = nnet::recurrent::{RECR_TYPE}<x_T, y_T, config_T>;
    # typedef {config_brnn_layer} BLAYER_CONFIG_{RECR_TYPE};
    # template<class x_T, class y_T, class config_T>
    # using backward_layer = nnet::recurrent::{RECR_TYPE}<x_T, y_T, config_T>;


#     static const unsigned n_in  = {n_in};
#     static const unsigned n_out = {n_out};
#     static const unsigned n_state = {n_state};
#     static const unsigned n_sequence = {n_sequence};
#     static const unsigned n_sequence_out = {n_sequence_out};
#     static const unsigned io_type = nnet::{strategy};
#     static const unsigned reuse_factor = {reuse};
#     static const bool store_weights_in_bram = false;

# }};\n"""




bidir_config_template = """struct config{index} : nnet::bidirectional_config {{
    typedef {accum_t.name} accum_t;
    typedef {forward_gru_gru_cell_2_weight_t.name} weight_t; 
    typedef {forward_gru_gru_cell_2_bias_t.name} bias_t; 

    typedef {config_frnn_layer} config_rnn_layer_f;
    typedef {config_brnn_layer} config_rnn_layer_b;

    typedef {config_mult_t1} mult_config1;
    typedef {config_mult_t2} mult_config2;
    typedef {recr_act_t} ACT_CONFIG_{RECR_TYPE};
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;
    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;

    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_state = {n_state};
    static const unsigned n_sequence = {n_sequence};
    static const unsigned n_sequence_out = {n_sequence_out};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned initial_state = {initial_state};
    static const bool store_weights_in_bram = false;

}};\n"""

#weights & bias add
bidir_function_template = 'nnet::bidirectional<{input_t}, {output_t}, {config}>({input}, {initial_state}, {output}, {wb_1}, {wrb_1}, {b_b}, {br_b},{wf_2},{wrf_2}, {b_f}, {br_f});'

bidirectional_include_list = ['nnet_utils/nnet_bidirectional.h', 'nnet_utils/nnet_recurrent.h']
print("template-----------------------------------------------")

class BidirectionalConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Bidirectional))
        self.template = bidir_config_template
        self.act_template = activ_birecr_config_template
        self.recr_act_template = birecr_activ_config_template
        self.mult1_template = birecr_mult_config_template
        self.mult2_template = birecr_mult_config_template
        self.flayer_template = bidir_recr_config_template
        self.blayer_template = bidir_recr_config_template
    
    def format(self, node):
        # bidirectional
        params = self._default_config_params(node)
        # print("params", params)
        params['RECR_TYPE'] = params['layer_type'].upper()
        params['recr_type'] = params['layer_type']
        params['config_frnn_layer'] = 'config{}_f'.format(node.index)
        params['config_brnn_layer'] = 'config{}_b'.format(node.index)
        params['config_mult_t1'] = 'config{}_1'.format(node.index)
        params['config_mult_t2'] = 'config{}_2'.format(node.index)
        params['recr_act_t'] = '{}_config{}_recr'.format(node.get_attr('recurrent_activation'), node.index)
        params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), node.index)
        params['n_state'] = params['n_out_forward']
        if(params['initial_state']==0):
            params['initial_state']==0

        if params['layer_type']=='gru':
            n_recr_mult = 3
        else: #LSTM
            n_recr_mult = 4

        params['n_in'] = node.get_input_variable().dim_names[1]
        params['n_sequence'] = node.get_input_variable().dim_names[0]
        # if node.get_attr('return_sequences'):
        #     params['n_sequence_out'] = node.get_output_variable().dim_names[0]
        #     params['n_state'] = node.get_output_variable().dim_names[1]
        #     params['n_out'] = node.get_output_variable().dim_names[1]
        # else:
        params['n_sequence_out'] = node.get_input_variable().dim_names[0]
            # params['n_state'] = node.get_output_variable().dim_names[0]
            # print(node.get_output_variable().dim_names[0])
        params['n_out'] = node.get_output_variable().dim_names[0]
        print("template----------------------------------------------check")
        # params['config_mult_t1'] = 'config{}_1'.format(node.index)
        # params['config_mult_t2'] = 'config{}_2'.format(node.index)
        # params['recr_act_t'] = '{}_config{}_recr'.format(node.get_attr('recurrent_activation'), node.index)
        # params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), node.index)
        params['strategy'] = node.get_attr('strategy')
        #params['static'] = 'true' if node.attributes['static'] else 'false'
       
       
        # params['recr_type'] = node.class_name.lower()
        # params['RECR_TYPE'] = node.class_name

        bidir_config = self.template.format(**params) # recr_config to bidir_config
        # forward layer
        flayer_params = self._default_config_params(node)
        blayer_params = self._default_config_params(node)
        # if flayer_params['layer_type']=='gru':
        n_recr_mult = 3
        # else: #LSTM
        #     n_recr_mult = 4
        flayer_params['direction'] = 'f'
        flayer_params['RECR_TYPE'] = flayer_params['layer_type'].upper()
        flayer_params['recr_type'] = flayer_params['layer_type']
        flayer_params['n_in'] = node.get_input_variable().dim_names[1]
        flayer_params['n_sequence'] = node.get_input_variable().dim_names[0]
        # if node.get_attr('return_sequences'):
        #     flayer_params['n_sequence_out'] = node.get_output_variable().dim_names[0]
        #     flayer_params['n_state'] = node.get_output_variable().dim_names[1]
        #     flayer_params['n_out'] = node.get_output_variable().dim_names[1]
        # else:
        flayer_params['n_sequence_out'] = node.get_input_variable().dim_names[0]
        flayer_params['n_state'] = params['n_out_forward']
        flayer_params['n_out'] = params['n_out_forward']

        flayer_params['config_mult_t1'] = 'config{}_1'.format(node.index)
        flayer_params['config_mult_t2'] = 'config{}_2'.format(node.index)
        flayer_params['recr_act_t'] = '{}_config{}_recr'.format(node.get_attr('recurrent_activation'), node.index)
        flayer_params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), node.index)
        flayer_params['strategy'] = node.get_attr('strategy')
        if(params['initial_state']==0):
            params['initial_state']==0

        flayer_config = self.flayer_template.format(**flayer_params)

        # backward layer 
        blayer_params['direction'] = 'b'
        # if blayer_params['layer_type']=='gru':
        n_recr_mult = 3
        # else: #LSTM
        #     n_recr_mult = 4
        blayer_params['RECR_TYPE'] = blayer_params['layer_type'].upper()
        blayer_params['recr_type'] = blayer_params['layer_type']
        blayer_params['n_in'] = node.get_input_variable().dim_names[1]
        blayer_params['n_sequence'] = node.get_input_variable().dim_names[0]
        # if node.get_attr('return_sequences'):
        #     blayer_params['n_sequence_out'] = node.get_output_variable().dim_names[0]
        #     blayer_params['n_state'] = node.get_output_variable().dim_names[1]
        #     blayer_params['n_out'] = node.get_output_variable().dim_names[1]
        # else:
        blayer_params['n_sequence_out'] = node.get_input_variable().dim_names[0]
        blayer_params['n_state'] = params['n_out_backward']
        blayer_params['n_out'] = params['n_out_backward']

        blayer_params['config_mult_t1'] = 'config{}_1'.format(node.index)
        blayer_params['config_mult_t2'] = 'config{}_2'.format(node.index)
        blayer_params['recr_act_t'] = '{}_config{}_recr'.format(node.get_attr('recurrent_activation'), node.index)
        blayer_params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), node.index)
        blayer_params['strategy'] = node.get_attr('strategy')
        if(params['initial_state']==0):
            params['initial_state']==0
        blayer_config = self.blayer_template.format(**blayer_params)
        ######################## correct modify under ######################################
        ####################################################################################

        act_params = self._default_config_params(node)
        recr_act_params = self._default_config_params(node)

        act_params['type'] = node.get_attr('activation')
        recr_act_params['type'] = node.get_attr('recurrent_activation')
        # if node.get_attr('return_sequences'):
        #     act_params['n_in'] = node.get_output_variable().dim_names[1]
        #     recr_act_params['n_in'] = node.get_output_variable().dim_names[1] + ' * %i'%(n_recr_mult-1)
        # else: # need to modify
        act_params['n_in'] = params['n_out_backward']
        # recr_act_params['n_in'] = params['n_out_backward'] + ' * %i'%(n_recr_mult-1)
        recr_act_params['n_in'] = params['n_out_backward'] * (n_recr_mult-1)
        # act_params['table_size'] = '1024'
        bi_act_config = self.act_template.format(**act_params)
        bi_recr_act_config = self.recr_act_template.format(**recr_act_params)

        mult_params1 = self._default_config_params(node)
        mult_params2 = self._default_config_params(node)

        mult_params1['n_in'] = node.get_input_variable().dim_names[1]
        # if node.get_attr('return_sequences'):
        #     mult_params1['n_out'] = node.get_output_variable().dim_names[1] + ' * %i'%n_recr_mult
        # else:
        #     # mult_params1['n_out'] = node.get_output_variable().dim_names[0] + ' * %i'%n_recr_mult
        #     mult_params1['n_out'] = params['n_out_backward'] + ' * %i'%n_recr_mult
        mult_params1['n_out'] = params['n_out_backward'] * n_recr_mult
        mult_params1['product_type'] = get_backend('vivado').product_type(node.get_input_variable().type.precision, node.get_weights('forward_gru_gru_cell_2_weight').type.precision)
        mult_params1['reuse'] = params['reuse']
        mult_params1['index'] = str(node.index) + '_1'
        mult_params1['nzeros'] = node.get_weights('forward_gru_gru_cell_2_weight').nzeros
        mult_params1['nonzeros'] = node.get_weights('forward_gru_gru_cell_2_weight').nonzeros
        # if node.get_attr('return_sequences'):
        #     mult_params2['n_in'] = node.get_output_variable().dim_names[1]
        #     mult_params2['n_out'] = node.get_output_variable().dim_names[1] + ' * %i'%n_recr_mult
        # else:
        #     # mult_params2['n_in'] = node.get_output_variable().dim_names[0]
        #     # mult_params2['n_out'] = node.get_output_variable().dim_names[0] + ' * %i'%n_recr_mult
        mult_params2['n_in'] = params['n_out_backward']
        # mult_params2['n_out'] = params['n_out_backward'] + ' * %i'%n_recr_mult
        mult_params2['n_out'] = params['n_out_backward'] * n_recr_mult
        mult_params2['product_type'] = get_backend('vivado').product_type(node.get_input_variable().type.precision, node.get_weights('forward_gru_gru_cell_2_recurrent_weight').type.precision)
        mult_params2['reuse'] = node.attributes['recurrent_reuse_factor']
        mult_params2['index'] = str(node.index) + '_2'
        mult_params2['nzeros'] = node.get_weights('forward_gru_gru_cell_2_recurrent_weight').nzeros
        mult_params2['nonzeros'] = node.get_weights('forward_gru_gru_cell_2_recurrent_weight').nonzeros

        bi_mult_config1 = self.mult1_template.format(**mult_params1)
        bi_mult_config2 = self.mult2_template.format(**mult_params2)


        print("template-----------------------------------------------1111")
    
        return bi_mult_config1 + '\n' + bi_mult_config2 + '\n' + bi_recr_act_config + '\n' + bi_act_config + '\n' + flayer_config + '\n' + blayer_config + '\n' + bidir_config # recr_config to bidir_config

class BidirectionalFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Bidirectional), include_header=bidirectional_include_list)
        self.template = bidir_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['initial_state'] = node.get_input_variable(node.inputs[0]).name
        params['wb_1'] = node.get_weights('backward_gru_1_gru_cell_1_weight').name
        params['wrb_1'] = node.get_weights('backward_gru_1_gru_cell_1_recurrent_weight').name
        params['b_b'] = node.get_weights('backward_gru_1_gru_cell_1_bias').name
        params['br_b'] = node.get_weights('backward_gru_recurrent_bias').name
        params['wf_2'] = node.get_weights('forward_gru_gru_cell_2_weight').name
        params['wrf_2'] = node.get_weights('forward_gru_gru_cell_2_recurrent_weight').name
        params['b_f'] = node.get_weights('forward_gru_gru_cell_2_bias').name
        params['br_f'] = node.get_weights('forward_gru_recurrent_bias').name
        params['activation'] = node.get_attr('activation')
        params['recurrent_activation'] = node.get_attr('recurrent_activation')
        params['recr_type'] = node.get_attr('layer_type')
  # pass params into function template
        print("template-----------------------------------------------pass")
        return self.template.format(**params)



                #mult 3& 4
        # mult_params3 = self._default_config_paramsS(node)
        # mult_params4 = self._default_config_params(node)

        # mult_params3['n_in'] = node.get_input_variable().dim_names[1]
        # if node.get_attr('return_sequences'):
        #     mult_params3['n_out'] = node.get_output_variable().dim_names[1] + ' * %i'%n_recr_mult
        # else:
        #     mult_params3['n_out'] = node.get_output_variable().dim_names[0] + ' * %i'%n_recr_mult
        # mult_params3['product_type'] = get_backend('vivado').product_type(node.get_input_variable().type.precision, node.get_weights('forward_gru_gru_cell_2_weight').type.precision)
        # mult_params3['reuse'] = params['reuse']
        # mult_params3['index'] = str(node.index) + '_3'
        # mult_params3['nzeros'] = node.get_weights('forward_gru_gru_cell_2_weight').nzeros
        # mult_params3['nonzeros'] = node.get_weights('forward_gru_gru_cell_2_weight').nonzeros
        # if node.get_attr('return_sequences'):
        #     mult_params4['n_in'] = node.get_output_variable().dim_names[1]
        #     mult_params4['n_out'] = node.get_output_variable().dim_names[1] + ' * %i'%n_recr_mult
        # else:
        #     mult_params4['n_in'] = node.get_output_variable().dim_names[0]
        #     mult_params4['n_out'] = node.get_output_variable().dim_names[0] + ' * %i'%n_recr_mult
        # mult_params4['product_type'] = get_backend('vivado').product_type(node.get_input_variable().type.precision, node.get_weights('forward_gru_gru_cell_2_recurrent_weight').type.precision)
        # mult_params4['reuse'] = node.attributes['recurrent_reuse_factor']
        # mult_params4['index'] = str(node.index) + '_4'
        # mult_params4['nzeros'] = node.get_weights('forward_gru_gru_cell_2_recurrent_weight').nzeros
        # mult_params4['nonzeros'] = node.get_weights('forward_gru_gru_cell_2_recurrent_weight').nonzeros

        # bi_mult_config3 = self.mult1_template.format(**mult_params3)
        # bi_mult_config4 = self.mult2_template.format(**mult_params4)