import theano
import numpy as np
import sys
from theano import tensor
from blocks.model import Model
from blocks.graph import ComputationGraph, apply_dropout
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp, Adam
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing, saveload, ProgressBar
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
from utils import get_stream, track_best, MainLoop
from model import nn_fprop
from config import config


# Load config parameters
locals().update(config)
# DATA
train_stream = get_stream(hdf5_file[network_mode], 'train', batch_size[network_mode])
test_stream = get_stream(hdf5_file[network_mode], 'test', batch_size[network_mode])

# MODEL
x = tensor.tensor3('features', dtype = 'floatX')
y = tensor.tensor3('targets', dtype = 'floatX')
x = x.swapaxes(0,1)
y = y.swapaxes(0,1)
# Required because Recurrent bricks receive as input [sequence, batch,
# features]

if network_mode == 0:
    cost, mu, sigma, mixing, output_hiddens, mu_linear, sigma_linear, mixing_linear, cells = nn_fprop(x, y, in_size[network_mode], out_size[network_mode], hidden_size[network_mode], num_layers, layer_models[network_mode][0], 'MDN', training=True)
elif network_mode == 1:
    y_hat, cost, cells = nn_fprop(x, y, in_size[network_mode], out_size[network_mode], hidden_size[network_mode], num_layers, layer_models[network_mode][0], 'SEC_MDN', training=True)
# COST
cg = ComputationGraph(cost)

if dropout[network_mode] > 0:
    # Apply dropout only to the non-recurrent inputs (Zaremba et al. 2015)
    inputs = VariableFilter(theano_name_regex=r'.*apply_input.*')(cg.variables)
    print inputs
    cg = apply_dropout(cg, inputs, dropout[network_mode])
    cost = cg.outputs[0]

# Learning algorithm
step_rules = [RMSProp(learning_rate=learning_rate[network_mode], decay_rate=decay_rate[network_mode]),
              StepClipping(step_clipping[network_mode])]
# step_rules = [Adam(learning_rate=learning_rate[network_mode]), StepClipping(step_clipping[network_mode])]
algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=CompositeRule(step_rules))

# Extensions
gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
step_norm = aggregation.mean(algorithm.total_step_norm)
monitored_vars = [cost, step_rules[0].learning_rate, gradient_norm, step_norm]

test_monitor = DataStreamMonitoring(variables=[cost], after_epoch=True,
                                   before_first_epoch=True, data_stream=test_stream, prefix="test")
train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_epoch=True,
                                       before_first_epoch=True, prefix='train')

# plot = Plot('Plotting example', channels=[['cost']], after_batch=True, open_browser=True)
extensions = [test_monitor, train_monitor, Timing(), Printing(after_epoch=True),
              FinishAfter(after_n_epochs=nepochs),
              saveload.Load(load_path),
#               saveload.Checkpoint(last_path,after_epoch=True),
              ] + track_best('test_cost', save_path[network_mode])

if learning_rate_decay[network_mode] not in (0, 1):
    extensions.append(SharedVariableModifier(step_rules[0].learning_rate,
                                             lambda n, lr: np.cast[theano.config.floatX](learning_rate_decay[network_mode] * lr), after_epoch=True, after_batch=False))

print 'number of parameters in the model: ' + str(tensor.sum([p.size for p in cg.parameters]).eval())
# Finally build the main loop and train the model
main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     model=Model(cost), extensions=extensions)
main_loop.run()
