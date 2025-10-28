import math
import sys

import torch
import numpy as np
# import matplotlib.pyplot as plt

# self_dis = 6.
y_band_width = 0.2
eps = 1e-10

def gaussian_kernel(dis, band_width):
    return (1 / (band_width * np.sqrt(2 * np.pi))) * torch.exp(-0.5 * (dis / band_width) ** 2)


def min_max_normalize(x, distance_mask):
    tensor = x.masked_fill(distance_mask, 9999.)
    min_val = torch.min(tensor, dim=-1).values

    tensor = x.masked_fill(distance_mask, -9999.)
    max_val = torch.max(tensor, dim=-1).values

    normalized_tensor = (tensor - min_val.unsqueeze(-1)) / (max_val - min_val + eps).unsqueeze(-1)  # 0
    normalized_tensor = normalized_tensor.masked_fill(distance_mask, 0.)

    return normalized_tensor

#
def distance(xyz, q, k, y_hat, distance_mask, big_inter_mask,
             t, band_width=6., std_att=True, self_distance=1):
    """
    :param xyz:
    :param q:
    :param k:
    :param y_hat:
    :param distance_mask:
    :param big_inter_mask:
    :param t:
    :param band_width:
    :param std_att:
    :param self_distance:
    :return:

    """
    #
    # if big_inter_mask is not None:
    #     big_inter_mask = big_inter_mask == 0
    #     # big_inter_mask = ~big_inter_mask
    #     distance_mask = big_inter_mask | distance_mask
    """"""
    xyz_dis = torch.cdist(xyz, xyz, p=2)
    I = torch.eye(xyz.size(0), dtype=xyz_dis.dtype, device=xyz_dis.device)
    # self_dis = band_width
    self_dis = band_width
    xyz_dis = xyz_dis + self_dis * I
    xyz_dis = xyz_dis.masked_fill(distance_mask, 9999.)
    xyz_score = gaussian_kernel(xyz_dis, band_width)
    # xyz_score = min_max_normalize(xyz_score, distance_mask)     # (0,1)
    """"""
    attn_score = q @ k.transpose(0, 1) / (t * math.sqrt(q.size(-1)))
    if torch.any(torch.isnan(attn_score)):
        print("attn is nan:", attn_score)
    # print(attn_score.max().item())
    # attn = torch.softmax(attn_score, dim=-1)
    # attn = attn.masked_fill(distance_mask, 0.)
    if std_att:
        attn_score = min_max_normalize(attn_score, distance_mask)   #

    attn_score = attn_score.masked_fill(distance_mask, -9999.)
    attn_score = torch.exp(attn_score)  #
    attn_score = (attn_score + attn_score.T) / 2
    # attn_score = torch.softmax(attn_score, dim=-1)
    if 0:
        # delta_y = torch.abs(y_hat - y_hat.transpose(0, 1))  # [0, 1]
        delta_y = y_hat
        delta_y = gaussian_kernel(delta_y, y_band_width)
        delta_y = delta_y.masked_fill(distance_mask, 0.)
        # delta_y = min_max_normalize(delta_y, distance_mask)

    #
    if big_inter_mask is not None:
        acc_para = 2
        inter_mask = big_inter_mask > 0
        inter_score = torch.where(inter_mask, acc_para, 1)
        final_score = xyz_score * attn_score * inter_score
    else:
        final_score = xyz_score * attn_score
        # final_score = xyz_score + attn_score
        #

    if torch.any(torch.isnan(final_score)):
        if torch.any(torch.isnan(xyz_score)):
            print("xyz is nan:", xyz_score)
        if torch.any(torch.isnan(attn_score)):
            print("attn is nan:", attn_score)
        sys.exit()

    return final_score


def mean_shift(beta, xyz, q, k, v, y_hat, distance_mask, big_inter_mask,
               t, band_width, max_iter, use_std_att, self_distance):
    max_iter = 1

    alpha = 1.0
    for _ in range(max_iter):

        kernel_weights = distance(xyz, q, k, y_hat, distance_mask, big_inter_mask, t, band_width, std_att=use_std_att, self_distance=self_distance)
        denominator = torch.sum(kernel_weights, dim=1, keepdim=True)
        kernel_weights = kernel_weights / denominator

        if torch.any(torch.isnan(kernel_weights)):
            print("kernel_weights is nan")
            kernel_weights = distance(xyz, q, k, y_hat, distance_mask, t, band_width)

            if torch.any(denominator == 0):
                print("denominator is 0")

        #
        new_xyz = kernel_weights @ xyz
        new_feature = kernel_weights @ v
        if torch.any(torch.isnan(new_feature)):
            print("new_feature is nan")

        new_xyz = (1 - beta) * xyz + beta * new_xyz
        # new_xyz = xyz
        new_feature = (1 - alpha) * v + alpha * new_feature

    return new_xyz, new_feature