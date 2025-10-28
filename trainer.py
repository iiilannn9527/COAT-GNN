import warnings

import numpy as np
import torch
import pickle
import esm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from data.dataset_final import Final_Dataset, sample, get_mask
from cluster_evaluation import evaluation, EarlyStopping, evaluation_5_fold
from readData import read_data
from tqdm import tqdm
import time
import os
import sys
from model.MS_model import GAT_MS_2_3, WeightedFocalLoss

warnings.filterwarnings('ignore')
parameters = {'lamda': 1.5, 'alpha': 0.7, 't': 1, 'band_width': 4., 'max_iter': 5, 'tol': 1e-2}
bw = 2


def train_console(train_tri, test_tri, two_mlp, use_l2_loss, output_path, device,
                         init_lr, batch_size, weight_decay, epochs,
                         beta=0.2, layers=8, iter_layers=4, embed_dim=256, heads=4, dropout=0.1, feature_dim=64,
                         reference_radius=14., self_distance=1., accumulation_steps=1, random_seed=42,
                         use_std_att=True, use_sample=False, use_std=False, use_inter=False, inter_i_graph=False, num_workers=0,
                  ):
    (dataset_name, train_path, train_seq_label, train_pse_pos, train_feature) = train_tri
    (_, test_path, test_seq_label, test_pse_pos, test_feature) = test_tri
    print(
        f"dataset: {test_path}"
        f"bata : {beta}, gat layers: {layers}, iter_layers: {iter_layers}, feature_dim: {feature_dim}, embed_dim: {embed_dim} \n"
        f"heads: {heads}, dropout: {dropout}, "
        f"batch size: {batch_size}, init lr: {init_lr}, epochs: {epochs},\n"
        f"graph_t_d: {reference_radius}, weight_decay: {weight_decay}"
    )
    para_set = str(beta) + '-' + str(layers) + '-' + str(iter_layers) + '-' + str(batch_size)
    if not os.path.exists(output_path + f"/{para_set}"):
        os.makedirs(output_path + f"/{para_set}")

    valid_seq_label, valid_pse_pos, valid_feature, valid_esm = None, None, None, None
    if dataset_name == 'GraphSet':
        pass
    elif dataset_name == 'PPBS' or dataset_name == 'BCE':
        valid_seq_label = test_path + 'valid_seq_label.txt'
        valid_pse_pos = test_path + 'valid_psepos.pkl'
        valid_feature = test_path + 'valid_node_feature.pkl'
        valid_esm = test_path + 'valid_esm_feature.pth'
        test_auroc_list = []
        test_auprc_list = []

    esm_dic = None

    parameters = {'lamda': 1.5, 'alpha': 0.7, 't': 1, 'band_width': 4., 'max_iter': 5, 'tol': 1e-2,
                  'self_distance': self_distance, 'reference_radius': reference_radius, 'beta': beta,
                  'use_std_att': use_std_att, 'use_std': use_std, 'use_inter': use_inter,
                  'inter_i_graph': inter_i_graph}

    train_key = pickle.load(open(train_pse_pos, 'rb')).keys()
    _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    auroc_list = []
    auprc_list = []
    test_result = {}
    for qq in range(0, 2):
        writer = {'train_BCE': SummaryWriter(f'{output_path}/{para_set}/{qq}'), }  #
        print(f'{qq} train')
        start_time = time.time()
        if dataset_name == 'PPBS' or dataset_name == 'BCE':
            test_auroc_max = 0
            test_auprc_max = 0
        auroc_max = 0
        auprc_max = 0
        model = GAT_MS_2_3(beta,
                           feature_dim,
                           embed_dim,
                           heads,
                           dropout=dropout,
                           gnn_layers=layers,
                           iter_layers=iter_layers,
                           two_mlp=two_mlp).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr,
                                      weight_decay=weight_decay if not use_l2_loss else 0)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=1e-5, threshold=1e-4, eps=1e-5, patience=5,
                                      verbose=True)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        early_stopping = EarlyStopping(patience=5, min_delta=0.0005, restore_best_weights=True)
        bio_learn_rate = True  #

        train_set, labels = read_data(train_seq_label, 0, 1000, train_key)
        proteins_id, strs, tokens = batch_converter(train_set)
        dataset = Final_Dataset(tokens, labels, proteins_id, train_feature, train_pse_pos, alphabet,
                                self_distance=self_distance, reference_radius=reference_radius, use_std=use_std,
                                use_inter=use_inter, inter_i_graph=inter_i_graph)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.
            norm_loss = 0.
            total_loss = 0.
            steps = 1
            step_loss = [0, 0, 0, 0, 0]
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                    num_workers=num_workers, collate_fn=dataset.batch_process)
            bar = tqdm(dataloader)
            optimizer.zero_grad()
            for i, data in enumerate(bar):
                batch_ids, batch_tokens, batch_labels, graphs, batch_feature, big_graph_dis, xyz, cums, big_inter_mask = data
                if next(model.parameters()).is_cuda:
                    batch_tokens = batch_tokens.to(device)
                    batch_labels = batch_labels.to(device)
                    graphs = graphs.to(device)
                    batch_feature = batch_feature.to(device)
                    big_graph_dis = big_graph_dis.to(device)
                    xyz = xyz.to(device)
                    if use_inter:
                        big_inter_mask = big_inter_mask.to(device)

                samples = []
                if use_sample:
                    samples = sample(batch_labels, device=device)
                true_index = get_mask(batch_tokens, alphabet)
                representations = None
                assert big_graph_dis.size(0) == xyz.size(0)
                # logit_1 is list of [logit0, logit1 ...]
                logit_1, logits, _ = model([batch_feature, representations], graphs, true_index, big_graph_dis, xyz,
                                           cums, big_inter_mask,
                                           return_loss=True, lamda=parameters['lamda'],
                                           alpha=parameters['alpha'],
                                           t=parameters['t'], band_width=parameters['band_width'],
                                           use_std_att=use_std_att, self_distance=self_distance)
                if use_sample:
                    temp_logits = torch.zeros((batch_size, batch_labels.size(-1), 2), dtype=logits.dtype).to(device)
                    temp_logits.reshape(-1, 2)[true_index] = logits
                    pre_logits = temp_logits.reshape(-1, 2)[samples]
                    loss = criterion(pre_logits, batch_labels.view(-1)[samples].squeeze())

                    temp_logits = torch.zeros((batch_size, batch_labels.size(-1), 2), dtype=logits.dtype).to(device)
                    temp_logits.reshape(-1, 2)[true_index] = logit_1
                    pre_logits = temp_logits.reshape(-1, 2)[samples]
                    loss_1 = criterion(pre_logits, batch_labels.view(-1)[samples].squeeze())
                else:
                    loss_ = []
                    for j in range(0, 1):
                        loss = criterion(logit_1[j], batch_labels.view(-1)[true_index])  # y hat
                        loss_.append(loss)
                        step_loss[j] += loss.item()
                    loss = criterion(logits, batch_labels.view(-1)[true_index])
                    loss_.append(loss)
                    loss_ = sum(loss_)
                regularization_loss = torch.tensor(0.0).to(loss.device)
                if use_l2_loss:
                    for param in model.parameters():
                        regularization_loss += param.pow(2.0).sum()
                    loss_sum = loss + regularization_loss * weight_decay
                else:
                    loss_sum = loss
                if torch.isnan(loss_sum) or torch.isinf(loss_sum):  #
                    print("Warning: Loss is NaN or Inf, resetting to 0")
                    loss_sum = torch.tensor(0.0, device=loss_sum.device)
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  #
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                # optimizer.step()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer['train_BCE'].add_scalar(f"Grad Norm/{name}", param.grad.norm().item(),
                                                   epoch * (len(dataloader)) + i)
                if len(samples) != 0 or ~use_sample:
                    running_loss += loss.item()
                    norm_loss += regularization_loss.item() * weight_decay
                    total_loss += loss_sum.item()
                steps += 1
                # break
            if (i + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            valid_loss, valid_temp, _ = evaluation('-', model, alphabet,
                                                   feature_path=valid_feature, file_path=valid_seq_label,
                                                   xyz_path=valid_pse_pos,
                                                   parameters=parameters, )
            print(f"epoch = {epoch}, sum loss = {total_loss / len(dataloader):.5f}, "
                  f"norm loss = {norm_loss / len(dataloader):.5f}, "
                  f"loss = {running_loss / len(dataloader):.5f}, valid_loss={valid_loss:.5f}")
            if auroc_max + auprc_max < valid_temp[8] + valid_temp[7]:
                auroc_max = valid_temp[8]
                auprc_max = valid_temp[7]
                torch.save(model.state_dict(), output_path + f"/{para_set}/model-{qq}.ckpt")
            if bio_learn_rate:
                if early_stopping(2 - valid_temp[7] - valid_temp[8], model) and optimizer.param_groups[0]['lr'] > 2.5e-5:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
                    print(f'learn rate descend to :', optimizer.param_groups[0]['lr'])

        print(f'auroc_max = {auroc_max:.5f}, auprc_max = {auprc_max:.5f}')
        auroc_list.append(auroc_max)
        auprc_list.append(auprc_max)
        end_time = time.time()
        print(f'time = {end_time - start_time:.1f}')

    print('$' * 100)
    print('!!!RESULT!!!')
    print(
        f"bata : {beta}, gat layers: {layers}, iter_layers: {iter_layers}, batch size: {batch_size * accumulation_steps}")
    print(auroc_list)
    print(auprc_list)
    auroc_np = np.array(auroc_list)
    arprc_np = np.array(auprc_list)
    print(f'auroc = {np.mean(auroc_np):.5f}, {np.std(auroc_np)}')
    print(f'auprc = {np.mean(arprc_np):.5f}, {np.std(arprc_np)}')
    print('$' * 100)

    test_result[para_set] = (
        (auroc_np, np.mean(auroc_np), np.std(auroc_np)), (arprc_np, np.mean(arprc_np), np.std(arprc_np)))

    return test_result


def train_console_2(train_tri, test_tri, two_mlp, use_l2_loss, output_path, device,
                    init_lr, batch_size, weight_decay, epochs,
                    beta=0.2, layers=8, iter_layers=4, embed_dim=256, heads=4, dropout=0.1, feature_dim=64,
                    reference_radius=14., self_distance=1., accumulation_steps=1, random_seed=42,
                    use_std_att=True, use_sample=False, use_std=False, use_inter=False, inter_i_graph=False, num_workers=0,
                    ):
    (dataset_name, train_path, train_seq_label, train_pse_pos, train_feature) = train_tri
    (_, test_path, test_seq_label, test_pse_pos, test_feature) = test_tri
    print(
        f"dataset: {test_path}"
        f"bata : {beta}, gat layers: {layers}, iter_layers: {iter_layers}, feature_dim: {feature_dim}, embed_dim: {embed_dim} \n"
        f"heads: {heads}, dropout: {dropout}, "
        f"batch size: {batch_size}, init lr: {init_lr}, epochs: {epochs},\n"
        f"graph_t_d: {reference_radius}, weight_decay: {weight_decay}"
    )
    para_set = str(beta) + '-' + str(layers) + '-' + str(iter_layers) + '-' + str(batch_size)
    if not os.path.exists(output_path + f"/{para_set}"):
        os.makedirs(output_path + f"/{para_set}")

    valid_seq_label, valid_pse_pos, valid_feature, valid_esm = None, None, None, None
    if dataset_name == 'GraphSet':
        pass
    elif dataset_name == 'PPBS' or dataset_name == 'BCE':
        valid_seq_label = test_path + 'valid_seq_label.txt'
        valid_pse_pos = test_path + 'valid_psepos.pkl'
        valid_feature = test_path + 'valid_node_feature.pkl'
        valid_esm = test_path + 'valid_esm_feature.pth'

    parameters = {'lamda': 1.5, 'alpha': 0.7, 't': 1, 'band_width': 4., 'max_iter': 5, 'tol': 1e-2,
                  'self_distance': self_distance, 'reference_radius': reference_radius, 'beta': beta,
                  'use_std_att': use_std_att, 'use_std': use_std, 'use_inter': use_inter,
                  'inter_i_graph': inter_i_graph}

    train_key = pickle.load(open(train_pse_pos, 'rb')).keys()
    _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    auroc_list = []
    auprc_list = []
    test_result = {}

    train_set, labels = read_data(train_seq_label, 0, 1000, train_key)
    proteins_id, strs, tokens = batch_converter(train_set)
    dataset = Final_Dataset(tokens, labels, proteins_id, train_feature, train_pse_pos, alphabet,
                            self_distance=self_distance, reference_radius=reference_radius, use_std=use_std,
                            use_inter=use_inter, inter_i_graph=inter_i_graph)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        val_dataloader = DataLoader(val_dataset, batch_size=30, shuffle=False, drop_last=False,
                                    num_workers=num_workers, collate_fn=dataset.batch_process)
        writer = {'train_BCE': SummaryWriter(f'{output_path}/{para_set}'), }
        print(f'{fold_num} train')
        start_time = time.time()
        auroc_max = 0
        auprc_max = 0
        model = GAT_MS_2_3(beta,
                           feature_dim,
                           embed_dim,
                           heads,
                           dropout=dropout,
                           gnn_layers=layers,
                           iter_layers=iter_layers,
                           two_mlp=two_mlp).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr,
                                      weight_decay=weight_decay if not use_l2_loss else 0)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=1e-5, threshold=1e-4, eps=1e-5,
                                      patience=5,
                                      verbose=True)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        early_stopping = EarlyStopping(patience=8, min_delta=0.0005, restore_best_weights=True)
        bio_learn_rate = True  #
        for epoch in range(epochs):
            model.train()
            running_loss = 0.
            norm_loss = 0.
            total_loss = 0.
            steps = 1
            step_loss = [0, 0, 0, 0, 0]
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                    num_workers=num_workers, collate_fn=dataset.batch_process)
            bar = tqdm(dataloader)
            optimizer.zero_grad()
            for i, data in enumerate(bar):
                batch_ids, batch_tokens, batch_labels, graphs, batch_feature, big_graph_dis, xyz, cums, big_inter_mask = data
                if next(model.parameters()).is_cuda:
                    batch_tokens = batch_tokens.to(device)
                    batch_labels = batch_labels.to(device)
                    graphs = graphs.to(device)
                    batch_feature = batch_feature.to(device)
                    big_graph_dis = big_graph_dis.to(device)
                    xyz = xyz.to(device)
                    if use_inter:
                        big_inter_mask = big_inter_mask.to(device)
                #
                samples = []
                if use_sample:
                    samples = sample(batch_labels, device=device)

                true_index = get_mask(batch_tokens, alphabet)

                representations = None
                assert big_graph_dis.size(0) == xyz.size(0)
                # logit_1 is list of [logit0, logit1 ...]
                logit_1, logits, _ = model([batch_feature, representations], graphs, true_index, big_graph_dis, xyz,
                                           cums, big_inter_mask,
                                           return_loss=True, lamda=parameters['lamda'],
                                           alpha=parameters['alpha'],
                                           t=parameters['t'], band_width=parameters['band_width'],
                                           use_std_att=use_std_att, self_distance=self_distance)
                if use_sample:
                    temp_logits = torch.zeros((batch_size, batch_labels.size(-1), 2), dtype=logits.dtype).to(device)
                    temp_logits.reshape(-1, 2)[true_index] = logits
                    pre_logits = temp_logits.reshape(-1, 2)[samples]
                    loss = criterion(pre_logits, batch_labels.view(-1)[samples].squeeze())

                    temp_logits = torch.zeros((batch_size, batch_labels.size(-1), 2), dtype=logits.dtype).to(device)
                    temp_logits.reshape(-1, 2)[true_index] = logit_1
                    pre_logits = temp_logits.reshape(-1, 2)[samples]
                    loss_1 = criterion(pre_logits, batch_labels.view(-1)[samples].squeeze())
                else:
                    loss_ = []
                    for j in range(0, 1):
                        loss = criterion(logit_1[j], batch_labels.view(-1)[true_index])  # y hat
                        loss_.append(loss)
                        step_loss[j] += loss.item()
                    loss = criterion(logits, batch_labels.view(-1)[true_index])
                    loss_.append(loss)
                    loss_ = sum(loss_)
                regularization_loss = torch.tensor(0.0).to(loss.device)
                if use_l2_loss:
                    for param in model.parameters():
                        regularization_loss += param.pow(2.0).sum()
                    loss_sum = loss + regularization_loss * weight_decay
                else:
                    loss_sum = loss
                if torch.isnan(loss_sum) or torch.isinf(loss_sum):  #
                    print("Warning: Loss is NaN or Inf, resetting to 0")
                    loss_sum = torch.tensor(0.0, device=loss_sum.device)
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  #
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                # optimizer.step()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer['train_BCE'].add_scalar(f"Grad Norm/{name}", param.grad.norm().item(),
                                                   epoch * (len(dataloader)) + i)
                if len(samples) != 0 or ~use_sample:
                    running_loss += loss.item()
                    norm_loss += regularization_loss.item() * weight_decay
                    total_loss += loss_sum.item()
                steps += 1
                # break
            if (i + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            valid_loss, valid_temp, _ = evaluation_5_fold('-', model, alphabet, dataset_name, valid_esm,
                                                          feature_path=valid_feature, file_path=valid_seq_label,
                                                          xyz_path=valid_pse_pos,
                                                          parameters=parameters,
                                                          dataloader=val_dataloader)
            print(f"epoch = {epoch}, sum loss = {total_loss / len(dataloader):.5f}, "
                  f"norm loss = {norm_loss / len(dataloader):.5f}, "
                  f"loss = {running_loss / len(dataloader):.5f}, valid_loss={valid_loss:.5f}")

            if auroc_max + auprc_max < valid_temp[8] + valid_temp[7]:
                auroc_max = valid_temp[8]
                auprc_max = valid_temp[7]
                torch.save(model.state_dict(), output_path + f"/{para_set}/model-{fold_num}.ckpt")
            if bio_learn_rate:
                if early_stopping(loss_sum.item(), model) and optimizer.param_groups[0]['lr'] > 2.5e-5:  #
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
                    print(f'learn rate descend to :', optimizer.param_groups[0]['lr'])

        print(f'auroc_max = {auroc_max:.5f}, auprc_max = {auprc_max:.5f}')
        auroc_list.append(auroc_max)
        auprc_list.append(auprc_max)

        end_time = time.time()
        print(f'time = {end_time - start_time:.1f}')
        break

    print('$' * 100)
    print('!!!RESULT!!!')
    print(f"bata : {beta}, gat layers: {layers}, iter_layers: {iter_layers}, batch size: {batch_size * accumulation_steps}")
    print(auroc_list)
    print(auprc_list)
    auroc_np = np.array(auroc_list)
    arprc_np = np.array(auprc_list)
    print(f'auroc = {np.mean(auroc_np):.5f}, {np.std(auroc_np)}')
    print(f'auprc = {np.mean(arprc_np):.5f}, {np.std(arprc_np)}')

    return test_result


def record_train(writer, epoch, dataset, data):
    writer['train_BCE'].add_scalar("model/train_loss", data[0], epoch)
    if dataset in ['PPBS', 'BCE']:
        writer['train_BCE'].add_scalar("model/valid_loss", data[1], epoch)
        writer['train_BCE'].add_scalar("model/valid_result", data[2], epoch)
    writer['train_BCE'].add_scalar("model/test_loss", data[3], epoch)
    writer['train_BCE'].add_scalar("model/test_result", data[4], epoch)
