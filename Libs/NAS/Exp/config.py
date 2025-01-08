from collections import namedtuple
from datetime import datetime

search_args = namedtuple('search_args', [
    'exp_name',
    'root',
    'dataset_root_path',
    'dataset',
    'batch_size',
    'learning_rate',
    'learning_rate_min',
    'momentum',
    'weight_decay',
    'report_freq',
    'gpu',
    'trials',
    'init_channels',
    'layers',
    'model_path',
    'cutout',
    'cutout_length',
    'drop_path_prob',
    'save',
    'seed',
    'grad_clip',
    'train_portion',
    'unrolled',
    'arch_learning_rate',
    'arch_weight_decay'
])

search_args.exp_name = 'HyperIDP'
search_args.root = '.'
search_args.dataset_root_path = './Input'  # root path of the data corpus
search_args.dataset_name = 'christiny'  # location of the data corpus

search_args.record_time = True  # used for run_with_record_time func
search_args.batch_size = 64  # batch size
search_args.learning_rate = 0.05  # init learning rate
search_args.learning_rate_min = 1e-3  # min learning rate
search_args.momentum = 0.9  # momentum
search_args.weight_decay = 5e-4  # weight decay
# search_args.gpu = gpu_id  # gpu device id
search_args.trials = 200  # num of trials
search_args.model_path = 'saved_models'  # path to save the model
# experiment name
search_args.save = f'{search_args.exp_name}-{search_args.dataset_name}'
search_args.seed = 3407  # random seed
search_args.grad_clip = 5  # gradient clipping
search_args.epsilon = 0.0  # the explore rate in the gradient descent process
search_args.train_portion = 0.5  # portion of training data
search_args.unrolled = False  # use one-step unrolled validation loss
search_args.arch_learning_rate = 3e-4  # learning rate for arch encoding
search_args.arch_weight_decay = 1e-3  # weight decay for arch encoding
search_args.transductive = True  # use transductive settings in train_search
search_args.with_conv_linear = False  # in NAMixOp with linear op
search_args.use_h_info = True
search_args.use_diff_g = False
search_args.fix_last = False  # fix last layer in design architectures
search_args.num_layers = 3  # num of aggregation layers

tune_args = namedtuple('tune_args', [
    'root',
    'exp_name',
    'dataset_root_path',
    'dataset_name',
    'arch_filename',
    'arch',
    'save',
    'num_layers',
    'seed',
    'grad_clip',
    'momentum',
    'learning_rate',
    'weight_decay',
    'tune_topK',
    'hidden_size',
    'in_dropout',
    'out_dropout',
    'activation',
    'record_time',
    'with_linear',
    'with_layernorm',
    'batch_size',
    'hyper_epoch',
    'epochs',
    'cos_lr',
    'fix_last'
])

tune_args.exp_name = 'HyperIDP'
tune_args.root = ''
tune_args.dataset_root_path = './Input'
tune_args.dataset_name = 'twitter'
tune_args.arch_filename = f'Output/Results/{tune_args.dataset_name}/Search/best_arch.txt'
tune_args.arch = ''
tune_args.save = 'logs/tune-{}_{}'.format(
    tune_args.dataset_name, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
tune_args.num_layers = 3
tune_args.seed = 3407
tune_args.grad_clip = 5
tune_args.momentum = 0.9
tune_args.learning_rate = 1e-3
tune_args.weight_decay = 5e-4
tune_args.tune_topK = False
tune_args.hidden_size = 64
tune_args.in_dropout = 0.3
tune_args.out_dropout = 0.3
tune_args.activation = 'relu'
tune_args.record_time = False
tune_args.with_linear = False
tune_args.with_layernorm = False
tune_args.batch_size = 256
tune_args.hyper_epoch = 50
tune_args.epochs = 400
tune_args.cos_lr = True
tune_args.fix_last = False
