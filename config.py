config = {}

config['network_mode'] = 0
config['batch_size'] = [128, 12]  # number of samples taken per each update. You might want to increase it to make the training faster, but you might not get the same result.
config['hidden_size'] = [200, 200]
config['learning_rate'] = [.0003, .0003]
config['learning_rate_decay'] = [0.999, 0.999]  # set to 0 to not decay learning rate
config['decay_rate'] = [0.999, 0.999]  # decay rate for rmsprop
config['step_clipping'] = [10.0, 10.0]  # clip norm of gradients at this value
config['dropout'] = [.0, .0]
config['nepochs'] = [1000, 1000]  # number of full passes through the training data
config['seq_length'] = [256, 24]  # number of waypoints in the truncated sequence
config['hdf5_file'] = ['input.hdf5', 'input_two.hdf5']  # hdf5 file with Fuel format
config['layer_models'] = [['lstm'], ['lstm']] # feedforward, lstm, rnn
config['num_layers'] = len(config['layer_models'][config['network_mode']])
config['input_columns'] = ['Pickup_longitude', 'Pickup_latitude']
config['output_columns'] = ['Pickup_longitude', 'Pickup_latitude']

config['in_size'] = [len(config['input_columns']), 200]
config['out_size'] = [len(config['output_columns']), 200]


# parameters of data augmentation
config['z_shift_range'] = [0,.5]
config['z_shift_offset'] = 2
config['x_shift_range'] = [0,.1]
config['x_shift_offset'] = 2
config['y_shift_range'] = [0,.3]
config['y_shift_offset'] = 2

# parameters about auxiliary predictions - The idea is to have some layers to predict another related prediction, for instance, predict the pose of object or the pose of gripper in next 4 timesteps
config['future_predictions'] = [1]
config['prediction_cost_weights'] = [1]

# parameters of multi-timescale learning : The idea is to enable different layers of LSTM or RNN to work at different time-scales
config['layer_resolutions'] = [1,1,1]
config['layer_execution_time_offset'] = [0,0,0]

# parameters of hierarchical model - The idea is to have a hierarchy of models, They are trained separately, and the input of the bottom model forms a part of its above modelconfig['hierarchy_resolutions'] = [ 8]
config['max_goal_difference'] = 1
config['level_name_in_hierarchy'] = 'bottom'  # top , middle , bottom
config['level_number_in_hierarchy'] = 0
config['hierarchy_resolutions'] = [8]

# parameters of multi-task learning : The idea is to train a network on data of multiple tasks. The ID if the task to be executed is given as an input in each time-step
config['multi_task_mode'] = 'ID'
config['game_tasks'] = [18]
config['trajs_paths'] = ['trajectories/18'] # 'trajectories/2', 'trajectories/18'
config['task_weights'] = [1]
config['task_waypoints_to_load'] = [400000]
config['waypoints_to_ignore'] = 0  # ignore number of waypoints from the beginning and the end of each file

# parameters of helper model - The idea is to firt train a helper network and then train another network that uses the hidden states of the helper network to predict its own output
config['use_helper_model'] = False
config['helper_models'] = ['models/ID_False_2-11-12-13-14-15-16-17_8_MDN_20_one-previous_True_True_1-1_0.0_1_bottom_False_2_20_5_50_best.pkl']
config['helper_hidden_size'] = 20
config['helper_num_layers'] = 2
config['helper_game_tasks'] = [2,11,12,13,14,15,16,17,18]

# We make a stacked RNN with the following skip connections added if the corresponding parameter is True
config['connect_x_to_h'] = True
config['connect_h_to_h'] = 'one-previous'  # all-previous , two-previous, one-previous
config['connect_h_to_o'] = True

# parameters of cost function
#config['cost_mode'] = ['MDN', 'SEC_MDN']  # MDN, MSE, Softmax
config['out_round_decimal'] = 2

# parameters of MDN
config['components_size'] = [100, 100]
config['seed'] = 66478

# outputting one dimension at a time parameters - Predicting only one dimension of the output at a time. The sequence length would be multiplied by the output dimension. Seems slow!
config['single_dim_out'] = False
config['single_dim_out_mode'] = 'each_layer_outputs_one_dim' # 'each_layer_outputs_one_dim'

# configs related to Baxter control
config['control_baxter'] = False

config['user_prefs'] = False

config['train_size'] = [0.80, 0.80]  # fraction of data that goes into train set
# path to the best model file
config['save_path'] = ['models/{0}_{1}_{2}_{3}_{4}_best.pkl'.format(config['components_size'][0], config['num_layers'], config['hidden_size'][0], config['batch_size'][0], config['seq_length'][0]), 'models/{0}_{1}_{2}_{3}_{4}_best.pkl'.format(config['components_size'][1], config['num_layers'], config['hidden_size'][1], config['batch_size'][1], config['seq_length'][1])]
# path to save the model of the last epoch
config['last_path'] = ['models/{0}_{1}_{2}_{3}_{4}_best.pkl'.format(config['components_size'][0], config['num_layers'], config['hidden_size'][0], config['batch_size'][0], config['seq_length'][0]), 'models/{0}_{1}_{2}_{3}_{4}_best.pkl'.format(config['components_size'][1], config['num_layers'], config['hidden_size'][1], config['batch_size'][1], config['seq_length'][1])]
config['load_path'] = config['save_path'][config['network_mode']]
config['hierarchy_models'] = [config['save_path'][config['network_mode']]]
