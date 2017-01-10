import theano
import numpy as np
from theano import tensor as T, shared
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from blocks import initialization
from blocks.bricks import Linear, Rectifier, cost
from blocks.bricks.parallel import Fork
from architectures import GatedRecurrent, LSTM, LN_LSTM, SimpleRecurrent
from blocks.bricks.cost import AbsoluteError, SquaredError
from config import config
from blocks.bricks import (MLP, Logistic, Initializable, FeedforwardSequence, Tanh,
                           NDimensionalSoftmax)
from blocks.initialization import Constant, Uniform
from blocks.utils import shared_like
import logging
locals().update(config)

def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()

def MDN_output_layer(x, h, y, y_mask, in_size, out_size, hidden_size, pred):
    if connect_h_to_o:
        hiddens = T.concatenate([hidden for hidden in h], axis=2)
        hidden_out_size = hidden_size * len(h)
    else:
        hiddens = h[-1]
        hidden_out_size = hidden_size

    #shared_hiddens = shared_like(hiddens[-1,:,:])
    mu_linear = Linear(name='mu_linear' + str(pred), input_dim=hidden_out_size, output_dim=out_size * components_size[network_mode])
    sigma_linear = Linear(name='sigma_linear' + str(pred), input_dim=hidden_out_size, output_dim=components_size[network_mode])
    mixing_linear = Linear(name='mixing_linear' + str(pred), input_dim=hidden_out_size, output_dim=components_size[network_mode])
    initialize([mu_linear, sigma_linear, mixing_linear])
    last_hiddens = hiddens[-1,:,:]
    mu = mu_linear.apply(last_hiddens)
    mu = mu.reshape((mu.shape[0], out_size, components_size[network_mode]))
    sigma = sigma_linear.apply(last_hiddens)
    sigma = T.nnet.softplus(sigma)
    mixing = mixing_linear.apply(last_hiddens)
    # apply softmax to mixing
    e_x = T.exp(mixing - mixing.max(axis=1, keepdims=True))
    mixing = e_x / e_x.sum(axis=1, keepdims=True)
    # calculate cost
    # nonzero_mask = (y.dimshuffle(0, 1, 2, 'x').round().clip(.1,.6)-.1)*2

    # y.shape = (seq, batch, features) mu.shape = (batch, features, component)    final shape (seq, batch, features, component)
    # shape of y_mask   (seq, 1)
    # y_mask = y_mask.reshape((y.shape[0], ))
    exponent = -0.5 * T.inv(sigma) * T.sum(((y.dimshuffle(0, 1, 2, 'x') - mu.dimshuffle('x', 0, 1, 2))) ** 2, axis=2)
    normalizer = (2 * np.pi * sigma)
    exponent = exponent + T.log(mixing) - (out_size * .5) * T.log(normalizer)
    # LogSumExp(x)

    # exponent =  y_mask.dimshuffle(0, 1, 'x').repeat(exponent.shape[2], axis=2)  * exponent
    max_exponent = T.max(exponent , axis=2, keepdims=True)
    mod_exponent = exponent - max_exponent
    gauss_mix = T.sum(T.exp(mod_exponent), axis=2, keepdims=True)
    log_gauss = T.log(gauss_mix) + max_exponent
    # mean over the batch, mean over sequence
    # cost = -(log_gauss * y_mask).sum() / y_mask.sum()
    log_gauss = y_mask.dimshuffle(0, 1, 'x') * log_gauss

    # cost = -T.mean(log_gauss, axis=1).mean()
    cost = - log_gauss.sum() / y_mask.sum()
    # cost = (cost * y_mask).sum() / y_mask.sum()
    output_hiddens = last_hiddens
    #shared_hiddens.set_value(hiddens[-1,:,:])
    return cost, mu, sigma, mixing, output_hiddens, mu_linear, sigma_linear, mixing_linear


def SEC_MDN_output_layer(x, h, y, in_size, out_size, hidden_size, pred):
    if connect_h_to_o:
        hiddens = T.concatenate([hidden for hidden in h], axis=2)
        hidden_out_size = hidden_size * len(h)
    else:
        hiddens = h[-1]
        hidden_out_size = hidden_size

    mu_linear = Linear(name = 'mu_linear' + str(pred), input_dim = hidden_out_size, output_dim = out_size * components_size[network_mode])
    sigma_linear = Linear(name = 'sigma_linear' + str(pred), input_dim = hidden_out_size, output_dim = components_size[network_mode])
    mixing_linear = Linear(name = 'mixing_linear' + str(pred), input_dim = hidden_out_size, output_dim = components_size[network_mode])
    initialize([mu_linear, sigma_linear, mixing_linear])
    mu = mu_linear.apply(hiddens)
    mu = mu.reshape((mu.shape[0], mu.shape[1], out_size, components_size[network_mode]))  # (seq, batch, 200, 50)
    sigma = sigma_linear.apply(hiddens)
    sigma = T.nnet.softplus(sigma)
    mixing = mixing_linear.apply(hiddens)
    # apply softmax to mixing
    e_x = T.exp(mixing - mixing.max(axis=2, keepdims=True))
    mixing = e_x / e_x.sum(axis=2, keepdims=True)
    # calculate cost
    exponent = -0.5 * T.inv(sigma) * T.sum((y.dimshuffle(0, 1, 2, 'x') - mu) ** 2, axis=2)
    normalizer = (2 * np.pi * sigma)
    exponent = exponent + T.log(mixing) - (out_size * .5) * T.log(normalizer)
    # LogSumExp(x)
    max_exponent = T.max(exponent , axis=2, keepdims=True)
    mod_exponent = exponent - max_exponent
    gauss_mix = T.sum(T.exp(mod_exponent), axis=2, keepdims=True)
    log_gauss = T.log(gauss_mix) + max_exponent
    # mean over the batch, mean over sequence
    cost = -T.mean(log_gauss, axis=1).mean()

    # sampling
    srng = RandomStreams(seed=seed)
    component = srng.multinomial(pvals=mixing)
    component_mean = T.sum(mu * component.dimshuffle(0, 1, 'x', 2), axis=3)
    component_std = T.sum(sigma * component, axis=2, keepdims=True)
    linear_output = srng.normal(avg=component_mean, std=component_std)
    linear_output.name = 'linear_output'

    # linear_output = 1
    return linear_output, cost


def MSE_output_layer(x, h, y, in_size, out_size, hidden_size, pred):
    if connect_h_to_o:
        hiddens = T.concatenate([hidden for hidden in h], axis=2)
        hidden_out_size = hidden_size * len(h)
        hidden_to_output = Linear(name='hidden_to_output' + str(pred), input_dim=hidden_out_size,
                                output_dim=out_size)
    else:
        hidden_to_output = Linear(name='hidden_to_output' + str(pred), input_dim=hidden_size,
                                output_dim=out_size)
        hiddens = h[-1]
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(hiddens)
    linear_output.name = 'linear_output'
    cost = T.sqr(y - linear_output).mean(axis=1).mean()  # + T.mul(T.sum(y[:,:,8:9],axis=1).mean(),2)
    return linear_output, cost

def softmax_output_layer(x, h, y, in_size, out_size, hidden_size, pred):
    if connect_h_to_o:
        hidden_to_output = Linear(name='hidden_to_output' + str(pred), input_dim=hidden_size * len(h),
                                output_dim=out_size)
        hiddens = T.concatenate([hidden for hidden in h], axis=2)
    else:
        hidden_to_output = Linear(name='hidden_to_output' + str(pred), input_dim=hidden_size,
                                output_dim=out_size)
        hiddens = h[-1]
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(hiddens)
    linear_output.name = 'linear_output'
    softmax = NDimensionalSoftmax()
    extra_ndim = 1 if single_dim_out else 2
    y_hat = softmax.apply(linear_output, extra_ndim=extra_ndim)
    cost = softmax.categorical_cross_entropy(y, linear_output, extra_ndim=extra_ndim).mean()

    return y_hat, cost

def output_layer(x, h, y, y_mask, in_size, out_size, hidden_size, cost_mode):
    hiddens = h
    if cost_mode == 'MDN':
        cost, mu, sigma, mixing, output_hiddens, mu_linear, sigma_linear, mixing_linear = MDN_output_layer(x, hiddens, y, y_mask, in_size, out_size, hidden_size, '-')
        cost.name = 'cost'
        return cost, mu, sigma, mixing, output_hiddens, mu_linear, sigma_linear, mixing_linear
    elif cost_mode == 'SEC_MDN':
        linear_output, cost = SEC_MDN_output_layer(x, hiddens, y, in_size, out_size, hidden_size, '-')
        cost.name = 'cost'
        return linear_output, cost
    elif cost_mode == 'MSE':
        linear_output, cost = MSE_output_layer(x, hiddens, y, in_size, out_size, hidden_size, '-')
    elif cost_mode == 'Softmax':
        linear_output, cost = softmax_output_layer(hiddens, y, in_size, out_size, hidden_size, '-')
    cost.name = 'cost'
    return linear_output, cost, mu, sigma, mixing, output_hiddens


def linear_layer(in_size, dim, x, h, n, first_layer=False):
    if first_layer:
        input = x
        linear = Linear(input_dim=in_size, output_dim=dim, name='feedforward' + str(n) + '-' )
    elif connect_x_to_h:
        input = T.concatenate([x] + [h[n - 1]], axis=1)
        linear = Linear(input_dim=in_size + dim, output_dim=dim, name='feedforward' + str(n) + '-' )
    else:
        input = h[n - 1]
        linear = Linear(input_dim=dim, output_dim=dim, name='feedforward' + str(n) + '-' )
    initialize([linear])
    return linear.apply(input)

def gru_layer(dim, h, n):
    fork = Fork(output_names=['linear' + str(n) + '-' , 'gates' + str(n) + '-' ],
                name='fork' + str(n) + '-' , input_dim=dim, output_dims=[dim, dim * 2])
    gru = GatedRecurrent(dim=dim, name='gru' + str(n) + '-' )
    initialize([fork, gru])
    linear, gates = fork.apply(h)
    return gru.apply(linear, gates)

def rnn_layer(in_size, dim, x, h, n, first_layer = False):
    if connect_h_to_h == 'all-previous':
        if first_layer:
            rnn_input = x
            linear = Linear(input_dim=in_size, output_dim=dim, name='linear' + str(n) + '-' )
        elif connect_x_to_h:
            rnn_input = T.concatenate([x] + [hidden for hidden in h], axis=2)
            linear = Linear(input_dim=in_size + dim * n, output_dim=dim, name='linear' + str(n) + '-' )
        else:
            rnn_input = T.concatenate([hidden for hidden in h], axis=2)
            linear = Linear(input_dim=dim * n, output_dim=dim, name='linear' + str(n) + '-' )
    elif connect_h_to_h == 'two-previous':
        if first_layer:
            rnn_input = x
            linear = Linear(input_dim=in_size, output_dim=dim, name='linear' + str(n) + '-' )
        elif connect_x_to_h:
            rnn_input = T.concatenate([x] + h[max(0, n - 2):n], axis=2)
            linear = Linear(input_dim=in_size + dim * 2 if n > 1 else in_size + dim, output_dim=dim, name='linear' + str(n) + '-' )
        else:
            rnn_input = T.concatenate(h[max(0, n - 2):n], axis=2)
            linear = Linear(input_dim=dim * 2 if n > 1 else dim, output_dim=dim, name='linear' + str(n) + '-' )
    elif connect_h_to_h == 'one-previous':
        if first_layer:
            rnn_input = x
            linear = Linear(input_dim=in_size, output_dim=dim, name='linear' + str(n) + '-' )
        elif connect_x_to_h:
            rnn_input = T.concatenate([x] + [h[n-1]], axis=2)
            linear = Linear(input_dim=in_size + dim, output_dim=dim, name='linear' + str(n) + '-' )
        else:
            rnn_input = h[n]
            linear = Linear(input_dim=dim, output_dim=dim, name='linear' + str(n) + '-' )
    rnn = SimpleRecurrent(dim=dim, activation=Tanh(), name=layer_models[n] + str(n) + '-' )
    initialize([linear, rnn])
    if layer_models[n] == 'rnn':
        return rnn.apply(linear.apply(rnn_input))
    elif layer_models[n] == 'mt_rnn':
        return rnn.apply(linear.apply(rnn_input), time_scale=layer_resolutions[n], time_offset=layer_execution_time_offset[n])

def lstm_layer(in_size, dim, x, x_mask, h, n, first_layer = False):
    if connect_h_to_h == 'all-previous':
        if first_layer:
            lstm_input = x
            linear = Linear(input_dim=in_size, output_dim=dim * 4, name='linear' + str(n) + '-' )
        elif connect_x_to_h:
            lstm_input = T.concatenate([x] + [hidden for hidden in h], axis=2)
            linear = Linear(input_dim=in_size + dim * (n), output_dim=dim * 4, name='linear' + str(n) + '-' )
        else:
            lstm_input = T.concatenate([hidden for hidden in h], axis=2)
            linear = Linear(input_dim=dim * (n + 1), output_dim=dim * 4, name='linear' + str(n) + '-' )
    elif connect_h_to_h == 'two-previous':
        if first_layer:
            lstm_input = x
            linear = Linear(input_dim=in_size, output_dim=dim * 4, name='linear' + str(n) + '-' )
        elif connect_x_to_h:
            lstm_input = T.concatenate([x] + h[max(0, n - 2):n], axis=2)
            linear = Linear(input_dim=in_size + dim * 2 if n > 1 else in_size + dim, output_dim=dim * 4, name='linear' + str(n) + '-' )
        else:
            lstm_input = T.concatenate(h[max(0, n - 2):n], axis=2)
            linear = Linear(input_dim=dim * 2 if n > 1 else dim, output_dim=dim * 4, name='linear' + str(n) + '-' )
    elif connect_h_to_h == 'one-previous':
        if first_layer:
            lstm_input = x
            linear = Linear(input_dim=in_size, output_dim=dim * 4, name='linear' + str(n) + '-' )
        elif connect_x_to_h:
            lstm_input = T.concatenate([x] + [h[n-1]], axis=2)
            linear = Linear(input_dim=in_size + dim, output_dim=dim * 4, name='linear' + str(n) + '-' )
        else:
            lstm_input = h[n-1]
            linear = LN_LSTM(input_dim=dim, output_dim=dim * 4, name='linear' + str(n) + '-' )
    lstm = LN_LSTM(dim=dim , name=layer_models[network_mode][n] + str(n) + '-' )
    initialize([linear, lstm])
    if layer_models[network_mode][n] == 'lstm':
        return lstm.apply(linear.apply(lstm_input), mask=x_mask)
    elif layer_models[network_mode][n] == 'mt_lstm':
        return lstm.apply(linear.apply(lstm_input), time_scale=layer_resolutions[n], time_offset=layer_execution_time_offset[n])

def add_layer(model, i, in_size, h_size, x, x_mask, h, cells, first_layer = False):
    if model == 'rnn' or model == 'mt_rnn':
        h.append(rnn_layer(in_size, h_size, x, h, i, first_layer))
    if model == 'gru':
        h.append(gru_layer(h_size, h[i], i))
    if model == 'lstm' or model == 'mt_lstm':
        state, cell = lstm_layer(in_size, h_size, x, x_mask, h, i, first_layer)
        h.append(state)
        cells.append(cell)
    if model == 'feedforward':
        h.append(linear_layer(in_size, h_size, x, h, i, first_layer))
    return h, cells
def nn_fprop(x, y, x_mask, y_mask, in_size, out_size, hidden_size, num_layers, model, cost_mode, training):
    if single_dim_out:
        out_size = 1
    cells = []
    h = []

    for i in range(num_layers):
        model = layer_models[network_mode][i]
        h, cells = add_layer(model, i, in_size, hidden_size, x, x_mask, h, cells, first_layer = True if i == 0 else False)
    return output_layer(x, h, y, y_mask, in_size, out_size, hidden_size, cost_mode) + (cells,)
