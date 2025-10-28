import pickle
import sys
import warnings
import torch
from torch import nn
from torch.utils.data import DataLoader
from data.dataset_final import Final_Dataset, get_mask, metric
from readData import read_data
import numpy as np

warnings.filterwarnings('ignore')

t = 0.35
graph_t_d = 10.
use_sample = 1


def sort_cluster(logits, label_matrix):
    protein_len = (label_matrix != -1).sum(-1)
    acc_len_sum = [0]
    for i in range(len(protein_len)):
        acc_len_sum.append(acc_len_sum[i] + protein_len[i])

    logits_batch = []
    for i in range(len(protein_len)):
        logits_batch.append(logits[acc_len_sum[i]: acc_len_sum[i + 1]])
        assert len(logits_batch[-1]) == protein_len[i]

    return logits_batch

def evaluation(mode: str, model, alphabet,
               feature_path,
               file_path=None,
               xyz_path=None,
               parameters=None,
               ):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    batch_converter = alphabet.get_batch_converter()
    esm_dic = None
    all_predictions = []
    all_labels = []

    pro_name = []
    pro_seq = []
    pro_logits = []
    true_labels = []

    position_layers = [[] for _ in range(model.iter_layers + 1)]
    feature_layers = [[] for _ in range(model.gnn_layers + model.iter_layers + 2)]
    name_seq_dic = {}

    times = 1

    model.eval()
    keys = pickle.load(open(xyz_path, 'rb')).keys()
    train_set, labels = read_data(file_path, 0, 1000, keys)
    proteins_id, strs, tokens = batch_converter(train_set)
    for i in range(len(proteins_id)):
        name_seq_dic[proteins_id[i]] = strs[i]

    dataset = Final_Dataset(tokens, labels, proteins_id, feature_path, xyz_path,
                            alphabet, self_distance=parameters['self_distance'],
                            reference_radius=parameters['reference_radius'],)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=0,
                            collate_fn=dataset.batch_process)
    # bar = tqdm(dataloader)
    loss = 0
    for i, data in enumerate(dataloader):
        times += 1
        # if mode is 'test':
        batch_ids, batch_tokens, batch_labels, graphs, batch_feature, big_graph_dis, xyz, cums, big_inter_mask = data

        pro_name.extend(batch_ids)
        for name in batch_ids:
            pro_seq.append(name_seq_dic[name])

        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            graphs = graphs.to(device)
            batch_feature = batch_feature.to(device)
            big_graph_dis = big_graph_dis.to(device)
            xyz = xyz.to(device)

        true_index = get_mask(batch_tokens, alphabet)
        with torch.no_grad():
            representations = None
            # if use_esm or dataset_name == 'DELPHI':
            #     representations = []
            #     for pro_id in batch_ids:
            #         representations.append(esm_dic[pro_id])
            #     representations = torch.stack(representations, dim=0)
            #     representations = representations.cuda()
            #     # batch_feature = representations
            logit_1, logits, (position, feature) = model([batch_feature, representations], graphs, true_index,
                                                         big_graph_dis, xyz, cums, big_inter_mask,
                                                         return_loss=True, lamda=parameters['lamda'],
                                                         alpha=parameters['alpha'],
                                                         t=parameters['t'], band_width=parameters['band_width'],
                                                         use_std_att=parameters['use_std_att'],
                                                         self_distance=parameters['self_distance'])
        logits = torch.softmax(logits, -1)
        valid_loss = criterion(logits, batch_labels.view(-1)[true_index])
        loss += valid_loss.item()
        probability, pred = torch.max(logits.data, -1)

        true_labels = batch_labels.reshape(-1)[true_index].squeeze().tolist()

        negative_pro, positive_pro = torch.chunk(logits, 2, -1)
        positive_chance = positive_pro.squeeze().tolist()

        if mode is 'test':
            batch_logits = sort_cluster(positive_chance, batch_labels)
            pro_logits.extend(batch_logits)
            # true_labels.extend()

            for j in range(len(position)):
                one_layer_pos = sort_cluster(position[j].cpu().numpy(), batch_labels)
                position_layers[j].extend(one_layer_pos)
            for j in range(len(feature)):
                one_layer_feat = sort_cluster(feature[j].cpu().numpy(), batch_labels)
                feature_layers[j].extend(one_layer_feat)
            # feature_layers[-2].extend(sort_cluster(feature[-2].cpu().numpy(), batch_labels))
            # feature_layers[-1].extend(sort_cluster(feature[-1].cpu().numpy(), batch_labels))

        all_labels.append(np.array(true_labels))
        all_predictions.append(np.array(positive_chance))

        # break
        # if times % 50 == 0:
        #     record = (pro_name, pro_seq, pro_logits, position_layers)
        #     with open("data/database-1/9982_FeatureAndPosition.pkl",
        #               "wb") as file:
        #         pickle.dump(record, file)
        #     file.close()
    loss = loss / len(dataloader)

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    _, temp, confusion_matrix_train = metric(all_predictions, all_labels, t)
    print("*" * 50 + f"AU-PRC = {temp[7]:.3f}, auc = {temp[8]:.3f}")


    record = (pro_name, pro_seq, pro_logits, position_layers, feature_layers)
    # with open("data/new_analyze_8224/9982_FeatureAndPosition.pkl", "wb") as file:
    #     pickle.dump(record, file)
    # file.close()
    # print("save data completed")

    return loss, temp, confusion_matrix_train


def evaluation_5_fold(mode: str, model, alphabet, dataset_name, esm_feature,
                      feature_path,
                      file_path=None,
                      xyz_path=None,
                      parameters=None,
                      dataloader=None,
                      ):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # elif dataset_name == 'DELPHI':
    #     esm_dic = torch.load(esm_feature)
    #     print("DELPHI feature load completed!!!")
    all_predictions = []
    all_labels = []

    pro_name = []
    pro_seq = []
    pro_logits = []
    true_labels = []

    position_layers = [[] for _ in range(model.iter_layers + 1)]
    feature_layers = [[] for _ in range(model.gnn_layers + model.iter_layers + 2)]
    name_seq_dic = {}

    times = 1

    model.eval()
    # keys = pickle.load(open(xyz_path, 'rb')).keys()
    # train_set, labels = read_data(file_path, 0, 1000, keys)
    # proteins_id, strs, tokens = batch_converter(train_set)
    # for i in range(len(proteins_id)):
    #     name_seq_dic[proteins_id[i]] = strs[i]
    #
    # dataset = Final_Dataset(tokens, labels, proteins_id, feature_path, xyz_path,
    #                         alphabet, self_distance=parameters['self_distance'],
    #                         reference_radius=parameters['reference_radius'],
    #                         use_std=parameters['use_std'], use_inter=parameters['use_inter'],
    #                         inter_i_graph=parameters['inter_i_graph'])
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=0,
    #                         collate_fn=dataset.batch_process)
    # bar = tqdm(dataloader)
    loss = 0
    for i, data in enumerate(dataloader):
        times += 1
        # if mode is 'test':
        batch_ids, batch_tokens, batch_labels, graphs, batch_feature, big_graph_dis, xyz, cums, big_inter_mask = data

        # pro_name.extend(batch_ids)
        # for name in batch_ids:
        #     pro_seq.append(name_seq_dic[name])

        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            graphs = graphs.to(device)
            batch_feature = batch_feature.to(device)
            big_graph_dis = big_graph_dis.to(device)
            xyz = xyz.to(device)
            if parameters['use_inter']:
                big_inter_mask = big_inter_mask.to(device)

        true_index = get_mask(batch_tokens, alphabet)
        with torch.no_grad():
            representations = None
            logit_1, logits, (position, feature) = model([batch_feature, None], graphs, true_index,
                                                         big_graph_dis, xyz, cums, big_inter_mask,
                                                         return_loss=True, lamda=parameters['lamda'],
                                                         alpha=parameters['alpha'],
                                                         t=parameters['t'], band_width=parameters['band_width'],
                                                         use_std_att=parameters['use_std_att'],
                                                         self_distance=parameters['self_distance'])
        logits = torch.softmax(logits, -1)
        valid_loss = criterion(logits, batch_labels.view(-1)[true_index])
        loss += valid_loss.item()
        probability, pred = torch.max(logits.data, -1)

        true_labels = batch_labels.reshape(-1)[true_index].squeeze().tolist()

        # true_pred = pred.reshape(-1)[locations].reshape(-1).tolist()

        negative_pro, positive_pro = torch.chunk(logits, 2, -1)
        positive_chance = positive_pro.squeeze().tolist()

        if mode is 'test':
            batch_logits = sort_cluster(positive_chance, batch_labels)
            pro_logits.extend(batch_logits)
            # true_labels.extend()

            for j in range(len(position)):
                one_layer_pos = sort_cluster(position[j].cpu().numpy(), batch_labels)
                position_layers[j].extend(one_layer_pos)
            for j in range(len(feature)):
                one_layer_feat = sort_cluster(feature[j].cpu().numpy(), batch_labels)
                feature_layers[j].extend(one_layer_feat)
            # feature_layers[-2].extend(sort_cluster(feature[-2].cpu().numpy(), batch_labels))
            # feature_layers[-1].extend(sort_cluster(feature[-1].cpu().numpy(), batch_labels))

        all_labels.append(np.array(true_labels))
        all_predictions.append(np.array(positive_chance))

        # break
        # if times % 50 == 0:
        #     record = (pro_name, pro_seq, pro_logits, position_layers)
        #     with open("data/database-1/9982_FeatureAndPosition.pkl",
        #               "wb") as file:
        #         pickle.dump(record, file)
        #     file.close()
    loss = loss / len(dataloader)

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    _, temp, confusion_matrix_train = metric(all_predictions, all_labels, t)
    print("*" * 50 + f"AU-PRC = {temp[7]:.3f}, auc = {temp[8]:.3f}")


    record = (pro_name, pro_seq, pro_logits, position_layers, feature_layers)
    # with open("data/new_analyze_8224/9982_FeatureAndPosition.pkl", "wb") as file:
    #     pickle.dump(record, file)
    # file.close()
    # print("save data completed")

    return loss, temp, confusion_matrix_train


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'wb', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """
        :param patience:
        :param min_delta:
        :param restore_best_weights:
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.best_model_weights = None
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_weights = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs, and reloading best model.")
            if self.restore_best_weights and self.best_model_weights:
                model.load_state_dict(self.best_model_weights)
            self.counter = 0
            return True

        return False
