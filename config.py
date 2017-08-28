config = {}

config['network_mode'] = 0
config['batch_size'] = [25, 1]  # number of samples taken per each update. You might want to increase it to make the training faster, but you might not get the same result.
config['hidden_size'] = [55, 70]
config['learning_rate'] = [.001, .001]
config['learning_rate_decay'] = [0.5, 0.999]  # set to 0 to not decay learning rate
config['decay_rate'] = [0.5, 0.999]  # decay rate for rmsprop
config['step_clipping'] = [5.0, 5.0]  # clip norm of gradients at this value
config['dropout'] = [.0, .0]
config['nepochs'] = [100000, 10000]  # number of full passes through the training data
config['num_features'] = 10 #3663 #3658 # 20: 6491 #5: 6500 # 60: 6488 # 10: 6494   #11894 + 78 #small 127
config['std'] = 50.0# 30: 7.0 # 20: 5.0 # 5: 2.0 # 60: 12.0 # 10: 3.0 # small 9.0
config['timestamp_length'] = 60  # in mins
config['seq_length'] = [168, 12]  # number of waypoints in the truncated sequence 1008
config['hdf5_file'] = ['input.hdf5', 'input_two.hdf5']  # hdf5 file with Fuel format
config['layer_models'] = [['lstm'], ['lstm']] # feedforward, lstm, rnn
config['num_layers'] = len(config['layer_models'][config['network_mode']])
# config['in_size'] = 10#2*config['num_features'] + 32
# config['out_size'] = 10

# We make a stacked RNN with the following skip connections added if the corresponding parameter is True
config['connect_x_to_h'] = True #False # True if it is LSTM
config['connect_h_to_h'] = 'one-previous'  # all-previous , two-previous, one-previous
config['connect_h_to_o'] = True #False # True if it is LSTM

# parameters of cost function
#config['cost_mode'] = ['MDN', 'SEC_MDN']  # MDN, MSE, Softmax
config['out_round_decimal'] = 2

# parameters of MDN
config['components_size'] = [10, 1]
config['seed'] = 66478
config['sampling_bias'] = 5.0

# outputting one dimension at a time parameters - Predicting only one dimension of the output at a time. The sequence length would be multiplied by the output dimension. Seems slow!
config['single_dim_out'] = False

config['user_prefs'] = False

config['train_size'] = [0.80, 0.80]  # fraction of data that goes into train set
# path to the best model file
config['save_path'] = ['models/{0}_{1}_{2}_{3}_{4}_best.pkl'.format(config['timestamp_length'], config['components_size'][0], config['num_layers'], config['hidden_size'][0], config['batch_size'][0]), 'models/{0}_{1}_{2}_{3}_best.pkl'.format(config['components_size'][1], config['num_layers'], config['hidden_size'][1], config['batch_size'][1])]
# path to save the model of the last epoch
config['last_path'] = ['models/{0}_{1}_{2}_{3}_{4}_last.pkl'.format(config['timestamp_length'], config['components_size'][0], config['num_layers'], config['hidden_size'][0], config['batch_size'][0]), 'models/{0}_{1}_{2}_{3}_last.pkl'.format(config['components_size'][1], config['num_layers'], config['hidden_size'][1], config['batch_size'][1])]
config['load_path'] = config['save_path'][config['network_mode']]
config['hierarchy_models'] = [config['save_path'][config['network_mode']]]
