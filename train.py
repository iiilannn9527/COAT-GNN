import pickle
import random
import torch
import numpy as np
from datasetSelect import DatasetControl, HyperParameterConfig
from model.MS_model import *
from cluster_evaluation import Logger
import time
import os
import sys
from trainer import train_console, train_console_2

warnings.filterwarnings('ignore')
#######################################################################################################################
# seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

controller = DatasetControl()
print("GraphSet, DELPHI, PPBS, BCE\n")
train_select = input("choose dataset")
train, co_test_dataset = controller.get_dataset_(train_select)
print(co_test_dataset)
test = co_test_dataset[input("choose test dataset")]
train_path = 'data/unity_data/' + train_select + '/' + train + '/'
test_path = 'data/unity_data/' + train_select + '/' + test + '/'

train_seq_label = train_path + 'seq_label.txt'
train_pse_pos = train_path + 'psepos.pkl'
train_feature = train_path + 'node_feature.pkl'
train_esm = train_path + 'esm_feature.pth'

test_seq_label = test_path + 'seq_label.txt'
test_pse_pos = test_path + 'psepos.pkl'
test_feature = test_path + 'node_feature.pkl'
test_esm = test_path + 'esm_feature.pth'

train_tri = (train_select, train_path, train_seq_label, train_pse_pos, train_feature)
test_tri = (train_select, test_path, test_seq_label, test_pse_pos, test_feature)

gpu_num = input("GPU, default 0\n")
device = torch.device('cuda:' + gpu_num)
#######################################################################################################################
output_path = rf'./trained_model/{train_select}/{test}/'
model_name = f'training'
time_path = time.strftime("%m-%d-%H-%M-%S", time.localtime())
print(time_path)
log_path = output_path + model_name
if not os.path.exists(log_path):
    os.makedirs(log_path)

# print(model_name)
#######################################################################################################################
_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
#######################################################################################################################
residue_psepos = pickle.load(open(train_pse_pos, 'rb'))
residue_psepos_dict = residue_psepos.keys()

Train = 1
TEST = 0

config = HyperParameterConfig()
params = config.get(train_select)

results = []

sys.stdout = Logger(os.path.normpath(log_path + f'/train_{train_select}.log'))

if train_select in ['PPBS', 'BCE']:
    test_result = train_console(train_tri, test_tri, params['two_mlp'], params['L2_norm'],
                                output_path, device, params['init_lr'], params['batch_size'],
                                params['weight_decay'], params['epochs'], beta=params['beta'],
                                layers=params['gat_layers'], iter_layers=params['num_layers'],
                                dropout=params['dropout'], feature_dim=params['feature_dim'])
elif train_select in ['GraphSet', 'DELPHI']:
    test_result = train_console_2(train_tri, test_tri, params['two_mlp'], params['L2_norm'],
                                  output_path, device, params['init_lr'], params['batch_size'],
                                  params['weight_decay'], params['epochs'], beta=params['beta'],
                                  layers=params['gat_layers'], iter_layers=params['num_layers'],
                                  dropout=params['dropout'], feature_dim=params['feature_dim'])
