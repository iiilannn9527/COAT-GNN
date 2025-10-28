from typing import Union
import esm
# from model.modules import *
from model.utils.mean_shift_cluster import mean_shift

import math
import dgl
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from dgl.nn import GATConv
import warnings

warnings.filterwarnings("ignore")


class GAT_MS(nn.Module):
    def __init__(self,
                 feat_dim,
                 in_features,
                 heads,
                 dropout=0.1,
                 edge_dim=2,
                 gnn_layers=4,
                 iter_layers=4,
                 qkv_bias=True,
                 augment_eps=0.1,
                 step_reward=False):
        super(GAT_MS, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = in_features
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.gnn_layers = gnn_layers
        self.iter_layers = iter_layers
        self.qkv_bias = qkv_bias
        self.augment_eps = augment_eps

        self.use_gat = True
        self.step_reward = step_reward

        self._init_submodules()

        print(f"model: \n"
              f"feature dim: {feat_dim}\n"
              f"hidden dim: {in_features}\n"
              f"attention heads: {heads}\n"
              f"gnn layers: {gnn_layers}\n"
              f"iter times: {iter_layers}\n"
              f"ust gat: {self.use_gat}\n"
              f"step reward: {step_reward}")

    def _init_submodules(self):
        use_plm = False
        self.fc_feature = nn.Linear(self.feat_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        if use_plm:
            self.plm_fc = nn.Linear(1280, self.hidden_dim)
            self.activation_plm = nn.ReLU()

        if self.use_gat:
            self.gat = GAT_Classifier(self.hidden_dim, self.heads, dropout=self.dropout, edge_dim=self.edge_dim,
                                      gnn_layers=self.gnn_layers)

        self.ms_cluster = MultiLayers_MS(self.iter_layers, self.hidden_dim, self.heads, dropout=self.dropout,
                                         bias=self.qkv_bias, step_reward=self.step_reward)

        self.classifier = nn.Linear(self.hidden_dim, 2)

    def forward(self, batch_feature, graph: dgl.DGLGraph, index_mask, big_graph_dis, xyz,
                return_loss, lamda, alpha,
                t, band_width, max_iter, tol):
        use_plm = False

        logit_ = []
        distance_mask = big_graph_dis > 99.

        if use_plm:
            str_feat = batch_feature[0]
            plm_feat = batch_feature[1]

            plm_feat = plm_feat.reshape(-1, plm_feat.size(-1))[index_mask]
            if self.training and self.augment_eps > 0:
                plm_feat = plm_feat + 0.1 * self.augment_eps * torch.randn_like(plm_feat)
            plm_feat = F.dropout(plm_feat, self.dropout, training=self.training)
            plm_feat = self.activation_plm(self.plm_fc(plm_feat))

            str_feat = str_feat.reshape(-1, str_feat.size(-1))[index_mask]
            if self.training and self.augment_eps > 0:
                str_feat = str_feat + 0.1 * self.augment_eps * torch.randn_like(str_feat)
            str_feat = F.dropout(str_feat, self.dropout, training=self.training)
            str_feat = self.activation(self.fc_feature(str_feat))

            x = str_feat + plm_feat

        else:
            x = batch_feature.reshape(-1, batch_feature.size(-1))[index_mask]

            # augment_eps
            if self.training and self.augment_eps > 0:
                x = x + 0.1 * self.augment_eps * torch.randn_like(x)

            x = F.dropout(x, self.dropout, training=self.training)

            x = self.activation(self.fc_feature(x))
            # x = self.fc_feature(x)

        if self.use_gat:
            #
            h0 = x
            logit, y_hat, x, gat_record = self.gat(x, graph, h0, lamda, alpha, return_loss)
            logit_.append(logit)
        else:
            y_hat = 0

        # mean-shift
        res = x
        x, record, logit = self.ms_cluster(xyz, x, y_hat, distance_mask, t, band_width)

        logit_.extend(logit)
        #
        x = res + x  #
        logit_f = self.classifier(x)

        position_re, feature_re = record
        gat_record.extend(feature_re[1:])

        if return_loss:
            return logit_, logit_f, (position_re, gat_record)
        else:
            return None, logit_f, (position_re, gat_record)


class GAT_Classifier(nn.Module):
    def __init__(self, in_features, heads, dropout=0.1, edge_dim=2, gnn_layers=4, use_norm=False):
        super(GAT_Classifier, self).__init__()
        self.hidden_dim = in_features
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.layers = gnn_layers
        self.use_norm = use_norm

        self.con_y = 0
        self.de_dim = 10

        self.gcnii_dim = 2 * self.hidden_dim

        self.gat_layers = nn.ModuleList(
            [
                GAT_layer(in_features, heads, dropout, edge_dim=edge_dim, use_norm=self.use_norm)
                for _ in range(self.layers)
            ]
        )

        self.classifier = nn.Linear(self.hidden_dim, 2)

        if self.con_y:
            self.decrease_dim = nn.Linear(self.hidden_dim, self.de_dim)
            self.con_classifier = nn.Linear(self.de_dim * 2, 2)

    def forward(self, x, graph, h0, lamda, alpha, return_loss, cums):
        #
        gat_record = [x]

        for layer_idx, layer in enumerate(self.gat_layers):
            x = layer(x, graph, h0, layer_idx + 1, cums, lamda, alpha)
            gat_record.append(x)

        if self.con_y:
            de_x = self.decrease_dim(x)
            de_x = torch.concat(
                [de_x.unsqueeze(1).repeat(1, de_x.size(0), 1), de_x.unsqueeze(0).repeat(de_x.size(0), 1, 1)], dim=-1)
            logit = self.con_classifier(de_x)
            probability = torch.softmax(logit, dim=-1)
            _, delta_y = torch.chunk(probability, 2, dim=-1)
            delta_y = delta_y.squeeze().detach()  # n*n
        else:
            logit = self.classifier(x)
            probability = torch.softmax(logit, dim=-1)
            _, y_hat = torch.chunk(probability, 2, dim=-1)
            y_hat = y_hat.detach()
            delta_y = torch.abs(y_hat - y_hat.transpose(0, 1))

        # if torch.any(torch.isnan(delta_y)):
        #     print("debug")

        if return_loss:
            return logit, delta_y, x, gat_record
        else:
            return None, delta_y, x, gat_record


class GAT_layer(nn.Module):
    def __init__(self, in_features, heads, dropout=0.1, edge_dim=0.2, use_norm=False):
        super(GAT_layer, self).__init__()
        self.hidden_dim = in_features
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.gcnii_dim = 2 * self.hidden_dim
        self.use_norm = use_norm

        self.gat = GATConv(in_features, int(in_features / heads), num_heads=heads, residual=False)
        self.weight = Parameter(torch.FloatTensor(self.gcnii_dim, self.hidden_dim))

        self.reset_parameters()

        if self.use_norm:
            self.norm_layer = GraphNormGAT(self.hidden_dim)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.weight.data.uniform_(-stdv, stdv)
        gain = nn.init.calculate_gain('relu')

    def forward(self, input, graph, h0, l, cums,
                lamda, alpha):
        x = self.gat(graph, input).view(input.size(0), -1)
        """GCNII"""
        theta = min(1, math.log(lamda / l + 1))
        support = torch.cat([x, h0], 1)
        r = (1 - alpha) * x + alpha * h0
        x = theta * torch.mm(support, self.weight) + (1 - theta) * r
        x = x + input

        if self.use_norm:
            x = self.norm_layer(x, cums)

        return x


#
class MultiLayers_MS(nn.Module):
    def __init__(self, use_norm, layers, hidden_dim, heads, dropout, bias, step_reward, same_space):
        super(MultiLayers_MS, self).__init__()
        self.layers_num = layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.bias = bias
        self.same_space = same_space
        self.use_norm = use_norm

        self.step_reward = step_reward

        self.ms_layers = nn.ModuleList(
            [
                MS(hidden_dim, heads, same_space, dropout, bias=bias, step_re=self.step_reward, use_norm=self.use_norm)
                for _ in range(self.layers_num)
            ]
        )
        if self.use_norm:
            self.norm_layer = GraphNormGAT(self.hidden_dim)

    def forward(self, xyz, x, y_hat, distance_mask, t, band_width, beta, cums, big_inter_mask, use_std_att, self_distance):
        if torch.any(torch.isnan(x)):
            print("x is nan:", x)
        position = [xyz]
        feature = [x]
        logit_ = []

        bw_list = [band_width] * self.layers_num

        for layer_idx, layer in enumerate(self.ms_layers):
            # res = x
            if self.step_reward:
                xyz, x, y_hat, logit = layer(xyz, x, y_hat, distance_mask, t, band_width[layer_idx], max_iter=1,
                                             tol=1e-4)
            else:
                xyz, x, _, logit = layer(beta, xyz, x, y_hat, distance_mask, cums, big_inter_mask,
                                         t, bw_list[layer_idx], 1, use_std_att, self_distance)
            # x = res + x
            logit_.append(logit)
            position.append(xyz)
            feature.append(x)

        return x, (position, feature), logit_


#
class MS(nn.Module):
    def __init__(self, hidden_dim, heads, same_space=False, dropout=0.1, bias=True, step_re=0, use_norm=False):
        super(MS, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.bias = bias
        self.dropout = dropout
        self.use_norm = use_norm

        self.step_reward = step_re

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)

        self.reset_parameters()

        # print(torch.isnan(self.out_proj.weight.data).any())

        if self.step_reward:
            self.classifier = nn.Linear(self.hidden_dim, 2)

        self.same_space = same_space
        if self.use_norm:
            self.norm_layer = GraphNormGAT(self.hidden_dim)
        # self.norm = nn.LayerNorm(self.hidden_dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim*4),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim*4, self.hidden_dim)
        # )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, beta, xyz, input, y_hat, distance_mask, cums, big_inter_mask,
                t, band_width, max_iter, use_std_att, self_distance):
        if torch.any(torch.isnan(self.out_proj.weight.data)):
            print("out_proj weight is nan:", self.q_proj.weight.data)
        if torch.any(torch.isnan(input)):
            print("input is nan:", input)

        res = input
        q = self.q_proj(input)
        if self.same_space:
            k = q
        else:
            k = self.k_proj(input)
        v = self.v_proj(input)
        xyz, x = mean_shift(beta, xyz, q, k, v, y_hat, distance_mask, big_inter_mask,
                            t, band_width, max_iter, use_std_att, self_distance)
        if torch.any(torch.isnan(x)):
            print("x is nan:", x)
        x = self.out_proj(x)
        x = res + x

        if self.use_norm:
            x = self.norm_layer(x, cums)

        #
        if self.step_reward:
            logit = self.classifier(x.detach())  #
            probability = torch.softmax(logit, dim=-1)
            _, y_hat = torch.chunk(probability, 2, dim=-1)
            delta_y = torch.abs(y_hat - y_hat.transpose(0, 1))  #
            delta_y = delta_y.detach()

            return xyz, x, delta_y, logit
        else:
            return xyz, x, 0, 0


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim=256, out_dim=2):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)  #

        #
        nn.init.xavier_uniform_(self.fc.weight)  #
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)  #


class RegularizedClassifier(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=32, out_dim=2, dropout=0.1):
        super(RegularizedClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)  #

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  #
        x = self.fc2(x)
        return x


class GAT_MS_2_3(nn.Module):
    def __init__(self,
                 beta,
                 feat_dim,
                 in_features,
                 heads,
                 dropout=0.1,
                 edge_dim=2,
                 gnn_layers=4,
                 iter_layers=4,
                 qkv_bias=True,
                 augment_eps=0.1,
                 step_reward=False,
                 use_esm=False,
                 two_mlp=True,
                 same_space=False,
                 use_norm=False,):
        super(GAT_MS_2_3, self).__init__()
        self.bata = beta
        self.feat_dim = feat_dim
        self.hidden_dim = in_features
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.gnn_layers = gnn_layers
        self.iter_layers = iter_layers
        self.qkv_bias = qkv_bias
        self.augment_eps = augment_eps

        self.use_gat = True
        self.step_reward = step_reward

        self.use_esm = use_esm
        self.use_complex_cla = two_mlp
        self.same_space = same_space
        self.use_norm = use_norm

        self._init_submodules()

        print(f"model: \n"
              f"feature dim: {feat_dim}\n"
              f"hidden dim: {in_features}\n"
              f"attention heads: {heads}\n"
              f"gnn layers: {gnn_layers}\n"
              f"iter times: {iter_layers}\n"
              f"ust gat: {self.use_gat}\n"
              )

    def _init_submodules(self):
        use_plm = False
        esm_dim = 1280
        self.fc_feature = nn.Linear(self.feat_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        # self.esm_fc = nn.Linear(1280, esm_dim)

        if self.use_gat:
            self.gat = GAT_Classifier(self.hidden_dim, self.heads, dropout=self.dropout, edge_dim=self.edge_dim, gnn_layers=self.gnn_layers,
                                      use_norm=self.use_norm)

        self.ms_cluster = MultiLayers_MS(self.use_norm, self.iter_layers, self.hidden_dim, self.heads, dropout=self.dropout,
                                         bias=self.qkv_bias, step_reward=self.step_reward, same_space=self.same_space,
                                         )

        if self.use_complex_cla:
            if self.use_esm:
                self.classifier = RegularizedClassifier(in_dim=esm_dim+self.hidden_dim, hidden_dim=32, out_dim=2, dropout=self.dropout)
            else:
                self.classifier = RegularizedClassifier(in_dim=self.hidden_dim, hidden_dim=32, out_dim=2, dropout=self.dropout)
        else:
            if self.use_esm:
                self.classifier = SimpleClassifier(in_dim=self.hidden_dim+esm_dim, out_dim=2)
            else:
                self.classifier = SimpleClassifier(in_dim=self.hidden_dim, out_dim=2)

    def forward(self, batch_feature: list, graph: dgl.DGLGraph, index_mask, big_graph_dis, xyz, cums, big_inter_mask,
                return_loss, lamda, alpha,
                t, band_width, use_std_att, self_distance):
        logit_ = []
        distance_mask = big_graph_dis >= 99.

        str_feat = batch_feature[0]
        x = str_feat.reshape(-1, str_feat.size(-1))[index_mask]
        if torch.any(torch.isnan(x)):
            print("x input x is nan:", x)
        # augment_eps
        # if self.training and self.augment_eps > 0:
        #     x = x + self.augment_eps * torch.randn_like(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.activation(self.fc_feature(x))
        if torch.any(torch.isnan(x)):
            print("fc x is nan:", x)
        if self.use_gat:
            #
            h0 = x
            logit, y_hat, x, gat_record = self.gat(x, graph, h0, lamda, alpha, return_loss, cums)
            logit_.append(logit)
        else:
            y_hat = 0
        """"""
        position_re = None
        if self.ms_cluster.layers_num != 0:
            res = x     # global res
            x, record, logit = self.ms_cluster(xyz, x, y_hat, distance_mask, t, band_width, self.bata, cums, big_inter_mask, use_std_att, self_distance)
            x = res + x
            logit_.extend(logit)

            position_re, feature_re = record
            gat_record.extend(feature_re[1:])
            # gat_record.append(x)
        """"""
        if self.use_esm:
            esm_feat = batch_feature[1]
            esm_feat = esm_feat.reshape(-1, esm_feat.size(-1))[index_mask]
            # if self.training and self.augment_eps > 0:
            #     esm_feat = esm_feat + self.augment_eps * torch.randn_like(esm_feat)
            esm_feat = F.dropout(esm_feat, self.dropout, training=self.training)

            f_res = esm_feat
            for i in range(self.iter_layers):
                res = esm_feat
                Q = esm_feat
                K = esm_feat
                V = esm_feat
                # 2. A = softmax(Q * K.T / sqrt(d))
                d_k = Q.size(-1)  # input_dim

                # (batch_size, seq_len, seq_len)
                attention_scores = torch.mm(Q, K.transpose(0, 1)) / d_k ** 0.5
                attention_scores = attention_scores.masked_fill(distance_mask, -9999.)

                # 3.
                #
                attention_weights = F.softmax(attention_scores, dim=-1)

                # 4. output = attention_weights * V
                esm_feat = torch.mm(attention_weights, V)  # (batch_size, seq_len, input_dim)
                esm_feat = (esm_feat + res) / 2

            esm_x = (f_res + esm_feat) / 2  #
            x = torch.concat([x, esm_x], dim=-1)

        logit_f = self.classifier(x)
        """"""
        # position_re, feature_re = record
        # gat_record.extend(feature_re[1:])
        gat_record.append(x)

        if return_loss:
            return logit_, logit_f, (position_re, gat_record)
        else:
            return None, logit_f, (position_re, gat_record)


class MultiLayers_MS_2_3(nn.Module):
    def __init__(self, layers, hidden_dim, heads, dropout, bias, step_reward):
        super(MultiLayers_MS_2_3, self).__init__()
        self.layers_num = layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.bias = bias

        self.step_reward = step_reward

        self.ms_layers = nn.ModuleList(
            [
                MS(hidden_dim, heads, dropout, bias=bias, step_re=self.step_reward)
                for _ in range(self.layers_num)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, xyz, x, y_hat, distance_mask,
                t, band_width):
        position = [xyz]
        feature = [x]
        logit_ = []

        band_width = [4, 4, 4, 4]

        for layer_idx, layer in enumerate(self.ms_layers):
            res = x

            if self.step_reward:
                xyz, x, y_hat, logit = layer(xyz, x, y_hat, distance_mask, t, band_width[layer_idx], max_iter=1,
                                             tol=1e-4)
            else:
                xyz, x, _, logit = layer(xyz, x, y_hat, distance_mask, t, band_width[layer_idx], max_iter=1, tol=1e-4)

            x[0] = res[0] + x[0]  #
            x[1] = res[1] + x[1]

            logit_.append(logit)
            position.append(xyz)
            feature.append(x)

        return x, (position, feature), logit_


class MS_2_3(nn.Module):
    def __init__(self, hidden_dim, heads, dropout=0.1, bias=True, step_re=0):
        super(MS_2_3, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.bias = bias
        self.dropout = dropout

        self.step_reward = step_re

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        """"""
        self.esm_q_proj = nn.Linear(64, 64, bias=bias)
        self.esm_k_proj = nn.Linear(64, 64, bias=bias)
        self.esm_v_proj = nn.Linear(64, 64, bias=bias)
        self.esm_out_proj = nn.Linear(64, 64, bias=bias)

        if self.step_reward:
            self.classifier = nn.Linear(self.hidden_dim, 2)
        # self.norm = nn.LayerNorm(self.hidden_dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim*4),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim*4, self.hidden_dim)
        # )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        """"""
        nn.init.xavier_uniform_(self.esm_q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.esm_k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.esm_v_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.esm_out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
            nn.init.constant_(self.esm_out_proj.bias, 0.0)

    def forward(self, xyz, input, y_hat, distance_mask,
                t, band_width, max_iter=5, tol=1e-4):
        str_feat = input[0]
        esm_feat = input[1]

        x = torch.concat(input, dim=-1)

        q = self.q_proj(str_feat)
        k = self.k_proj(str_feat)
        v = self.v_proj(str_feat)

        esm_q = self.esm_q_proj(esm_feat)
        esm_k = self.esm_k_proj(esm_feat)
        esm_v = self.esm_v_proj(esm_feat)

        # xyz, x = mean_shift(xyz, q, k, v, y_hat, distance_mask,
        #                     t, band_width, max_iter)
        # x = self.out_proj(x)
        """"""
        xyz, x = mean_shift(xyz, q, k, v, y_hat, distance_mask, t, band_width, max_iter)
        x = self.out_proj(x)

        #
        if self.step_reward:
            logit = self.classifier(x.detach())  #
            probability = torch.softmax(logit, dim=-1)
            _, y_hat = torch.chunk(probability, 2, dim=-1)
            delta_y = torch.abs(y_hat - y_hat.transpose(0, 1))  #
            delta_y = delta_y.detach()

            return xyz, x, delta_y, logit
        else:
            return xyz, x, 0, 0


class GraphNormGAT(nn.Module):
    def __init__(self, in_features):
        super(GraphNormGAT, self).__init__()
        #
        self.gamma = nn.Parameter(torch.ones(1, in_features))
        self.beta = nn.Parameter(torch.zeros(1, in_features))

    def forward(self, x, batch:list):
        tem_x = torch.zeros_like(x).to(x.device)
        for i in range(len(batch)-1):
            data = x[batch[i]:batch[i+1]]

            #
            mean = data.mean(dim=0, keepdim=True)
            std = data.std(dim=0, keepdim=True)

            #
            normalized_data = (data - mean) / (std + 1e-5)

            #
            out = self.gamma * normalized_data + self.beta

            tem_x[batch[i]:batch[i+1], :] = out

        return tem_x


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()