import copy
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import dgl
import math
import numpy as np
# from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

label_pad = -1
MAP_CUTOFF = 14
DIST_NORM = 15

acid = torch.tensor([13, 9, ])
alkali = torch.tensor([21, 15, 10, ])
hydrophobic = torch.tensor([18, 5, 4, 20, 12, 22, 14, 7, ])
hydrogen = torch.tensor([10, 15, 13, 9, 8, 17, 11, 16, 21, 19, 22, ])


def index2onehot(seq_token):
    inter_code = torch.zeros_like(seq_token).type('torch.FloatTensor').unsqueeze(-1).repeat(1, 4)
    inter_code[:, 0][torch.isin(seq_token, acid)] = 1.
    inter_code[:, 1][torch.isin(seq_token, alkali)] = 1.
    inter_code[:, 2][torch.isin(seq_token, hydrophobic)] = 1.
    inter_code[:, 3][torch.isin(seq_token, hydrogen)] = 1.

    return inter_code


class Final_Dataset(Dataset):
    def __init__(self, tokens, labels, protein_ids, feature_path, xyz_path,
                 alphabet, self_distance=1., reference_radius=14., use_std=False, use_inter=False, inter_i_graph=False):
        super().__init__()
        self.tokens = tokens  # tensor
        self.labels = labels  # list
        self.protein_ids = protein_ids
        self.alphabet = alphabet
        self.len = tokens.size(0)
        self.max_len = tokens.size(-1)

        self.xyz = pickle.load(open(xyz_path, 'rb'))
        # ext = Path(feature_path).suffix
        # self.feature = pickle.load(open(feature_path, 'rb')) if ext == '.pkl' else torch.load(feature_path)
        self.feature = pickle.load(open(feature_path, 'rb'))

        #
        self.reference_radius = reference_radius  #
        self.self_distance = self_distance  #
        self.need_edge = None
        self.need_std = use_std  #
        self.use_inter = use_inter  #
        self.i_i_g = inter_i_graph
        if not self.use_inter:
            self.i_i_g = False  #

    def __getitem__(self, item):
        protein_key = self.protein_ids[item]

        token = self.tokens[item]
        label = copy.deepcopy(self.labels[item])

        inter_mask = None
        if self.use_inter:
            inter_tensor = index2onehot(token)
            inter_mask = torch.matmul(inter_tensor, inter_tensor.transpose(-1, -2))
            inter_mask = inter_mask[1:1 + len(label), 1:1 + len(label)]

        xyz = torch.from_numpy(self.xyz[protein_key]).to(torch.float32)  # tensor

        if torch.is_tensor(self.feature.get(protein_key)):
            node_features = self.feature.get(protein_key).to(torch.float32)  # tensor
        else:
            node_features = torch.from_numpy(self.feature.get(protein_key)).to(torch.float32)  # tensor
        #
        # reference_res_psepos = xyz[0]
        # pos = xyz - reference_res_psepos
        # node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / 15], dim=-1)
        if self.need_std:
            node_features = self.z_score(node_features)
        if torch.any(torch.isnan(node_features)):
            print(f"{protein_key} feature is nan:", node_features)
        #
        #
        distance_matrix = torch.cdist(xyz, xyz, p=2)
        I = torch.eye(xyz.size(0), dtype=distance_matrix.dtype, device=distance_matrix.device)
        distance_matrix = distance_matrix + self.self_distance * I

        radius_index_list = self.cal_edges(distance_matrix.numpy(), self.reference_radius, inter_mask)
        graph = dgl.graph((radius_index_list[0], radius_index_list[1]), num_nodes=len(label), idtype=torch.int32)
        # graph = dgl.add_self_loop(graph)
        if self.need_edge:
            edge_feat = self.cal_edge_attr(radius_index_list, xyz)
            edge_feat = np.transpose(edge_feat, (1, 2, 0))
            edge_feat = edge_feat.squeeze(1)
            graph.edata['ex'] = torch.tensor(edge_feat, dtype=torch.float32)

        return protein_key, token, label, graph, distance_matrix, node_features, xyz, inter_mask

    def __len__(self):
        return self.len

    def batch_process(self, samples):
        protein_keys, batch_tokens, batch_labels, batch_graphs, list_distance_matrix, batch_feature, xyz, inter_mask = map(
            list, zip(*samples))

        max_len = self.max_len
        big_graph_size = 0
        cums = [0]
        for dis in list_distance_matrix:
            big_graph_size += dis.size(0)
            cums.append(big_graph_size)

        big_graph_dis = torch.ones((big_graph_size, big_graph_size), dtype=list_distance_matrix[0].dtype) * 9999.
        big_inter_mask = None
        if self.use_inter:
            big_inter_mask = torch.zeros_like(big_graph_dis)  # float 0/1
        for i, dis in enumerate(list_distance_matrix):
            assert dis.size(0) <= max_len - 2
            big_graph_dis[cums[i]:cums[i + 1], cums[i]:cums[i + 1]] = dis
            if self.use_inter:
                big_inter_mask[cums[i]:cums[i + 1], cums[i]:cums[i + 1]] = inter_mask[i]

        pad_feature = torch.zeros((len(batch_feature), len(batch_tokens[0]), batch_feature[0].size(-1)),
                                  dtype=batch_feature[0].dtype)
        for i, label in enumerate(batch_labels):
            if len(label) < max_len:
                label.extend([label_pad] * (max_len - len(label) - 1))
                batch_labels[i] = [int(-1)] + label
            pad_feature[i][1:batch_feature[i].size(0) + 1] = batch_feature[i]

        batch_tokens = [s.tolist() for s in batch_tokens]
        batch_tokens = torch.LongTensor(batch_tokens)
        batch_labels = torch.LongTensor(batch_labels)

        batch_graphs = dgl.batch(batch_graphs)

        # xyz_l = []
        # for i in range(len(xyz)):
        #     xyz_l.extend(xyz[i])
        # xyz_l = torch.tensor(xyz_l, dtype=torch.float32)
        xyz = torch.cat(xyz, dim=0).to(dtype=torch.float32)

        return protein_keys, batch_tokens, batch_labels, batch_graphs, pad_feature, big_graph_dis, xyz, cums, big_inter_mask

    def init_inter_mask(self):
        for key in self.xyz:
            token = self.tokens[key]
            inter_tensor = index2onehot(token)
            inter_mask = torch.matmul(inter_tensor, inter_tensor.transpose(-1, -2))
            self.inter_mask_dict[key] = inter_mask

    def cal_edges(self, distance_matrix, radius, inter_mask):  # to get the index of the edges
        mask = ((distance_matrix >= 0) * (distance_matrix <= radius))
        if self.i_i_g:
            inter_mask = inter_mask > 0
            mask = mask * inter_mask.numpy()
            I = np.eye(mask.shape[0], dtype=bool)  #
            mask = mask | I
        adjacency_matrix = mask.astype(np.int)
        radius_index_list = np.where(adjacency_matrix == 1)
        radius_index_list = [list(nodes) for nodes in radius_index_list]

        return radius_index_list

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])

        return radius_attr_list

    def z_score(self, sasa):
        cos_mean = torch.mean(sasa, dim=0)
        sigma = torch.sqrt(torch.mean((sasa - cos_mean.unsqueeze(0)) ** 2, dim=0))

        sasa = (sasa - cos_mean.unsqueeze(0)) / (sigma.unsqueeze(0) + 1e-5)
        sasa = sasa.masked_fill(sasa.isnan(), 0.)

        return sasa


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def graph_collate(samples):
    sequence_name, label, node_features, adj_sc, G, adj_ca = map(list, zip(*samples))
    label = torch.Tensor(np.array(label))
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_sc = torch.Tensor(np.array(adj_sc))
    adj_ca = torch.Tensor(np.array(adj_ca))
    return sequence_name, label, node_features, adj_sc, G_batch, adj_ca

def get_mask(tokens, alphabet):
    """

    :rtype: object
    """
    mask = (tokens == alphabet.cls_idx) | (tokens == alphabet.eos_idx) | (tokens == alphabet.padding_idx)
    # tokens = tokens * ~mask
    true_index = torch.nonzero((~mask).reshape(-1))

    return true_index.squeeze().tolist()


def metric(preds, labels, t=0.5):
    """

    :param preds:
    :param labels:
    :param t:
    :return:
    """
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)

    assert len(labels) == len(preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds > t).ravel()
    # tn, fp, fn, tp = confusion_matrix(labels, preds > 0).ravel()
    # sn = tp / (tp + fn)
    # sp = tn / (tn + fp)
    # pr = tp / (tp + fp)

    acc = (tn + tp) / (tn + fn + fp + tp)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    zero_pre = tn / (tn + fn)
    zero_rec = tn / (tn + fp)
    f1 = 2 * pre * rec / (pre + rec)
    mcc = (tp * tn - fp * fn) / (
                math.sqrt((tp + fp)) * math.sqrt((tp + fn)) * math.sqrt((tn + fp)) * math.sqrt((tn + fn)))
    try:
        AUROC = roc_auc_score(labels, preds)
        precisions, recalls, _ = precision_recall_curve(labels, preds)  #######
        AUPRC = auc(recalls, precisions)
    except ValueError as v:
        print("prediction value too low")
        AUROC = 0.
        AUPRC = 0.

    return 'pad', [acc, pre, rec, zero_pre, zero_rec, f1, mcc, AUPRC, AUROC], [[tn, fp], [fn, tp]]

    # #
    # precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    # f1_scores_pr = 2 * (precision * recall) / (precision + recall)
    # best_threshold_pr = thresholds[np.argmax(f1_scores_pr)]


def sample(labels, device):
    """"""
    sample_rate = 1
    location = torch.nonzero(labels.reshape(-1) * (labels.reshape(-1) == 1)).to(device)
    sample_number = int(location.size(0) * sample_rate)
    negative_index = torch.nonzero(torch.ones_like(labels).reshape(-1) * (labels.reshape(-1) == 0)).tolist()
    random.shuffle(negative_index)
    negative_samples_index = negative_index[:sample_number]
    samples = negative_samples_index + location.tolist()
    samples_1_list = [s[0] for s in samples]

    # return samples
    return samples_1_list