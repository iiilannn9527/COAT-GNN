import random
import numpy as np
from datasetSelect import DatasetControl, HyperParameterConfig
from model.MS_model import *
from cluster_evaluation import Logger, evaluation


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


reference_radius = 14.
self_distance = 1.
_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
config = HyperParameterConfig()
params = config.get(train_select)
model = GAT_MS_2_3(params['beta'],
                   params['feature_dim'],
                   params['embed_dim'],
                   4,
                   dropout=0.2,
                   gnn_layers=params['gat_layers'],
                   two_mlp=params['two_mlp'],
                   iter_layers=params['num_layers']).to(device)

key = train_select + '-' + test
trained_model_dic = {
    'GraphSet-test_60': 'GraphSet',
    'GraphSet-test_287': 'GraphSet',
    'DELPHI-test': 'DELPHI_SET',
    'PPBS-test_70': 'PPBS-T_70',
    'PPBS-test_homo': 'PPBS-T_70',
    'PPBS-test_none': 'PPBS-T_70',
    'PPBS-test_topo': 'PPBS-T_70',
    'BCE-test': 'BCE'
}

model_state = trained_model_dic[key]
state_dict = torch.load(f'trained_model/{model_state}.ckpt')
model.load_state_dict(state_dict)
parameters = {'lamda': 1.5, 'alpha': 0.7, 't': 1, 'band_width': 4., 'max_iter': 5, 'tol': 1e-2,
              'self_distance': self_distance, 'reference_radius': reference_radius, 'beta': params['two_mlp'],
              'use_std_att': True,}


valid_loss, temp, _ = evaluation('none', model, alphabet,
                                 feature_path=test_feature, file_path=test_seq_label,
                                 xyz_path=test_pse_pos,
                                 parameters=parameters,
                                 )
print(f"AU-PRC = {temp[7]:.3f}, auc = {temp[8]:.3f}")