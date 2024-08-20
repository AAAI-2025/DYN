import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
)
import torchvision.models as models
import time
import argparse
import numpy as np
import torch
import scipy.sparse as sp
import warnings
import os

class ClusterAwareBatchNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1, affine=True):
        super(ClusterAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self._bn = nn.BatchNorm2d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )
        self.source_rate = nn.parameter.Parameter(torch.tensor(0.8))
        # self.test_rate = nn.parameter.Parameter(torch.tensor(0.0))
        # self.Domain_discriminator = False

    
    def Lisc(self, x, mu_b, sigma2_b, source_rate):
        b, c, _, _ = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)
        data = mu.view(b, c)
        c1, _ = FINCH(data)
        unique_labels, inverse_indices = torch.unique(c1, dim=0, return_inverse=True)
        print('[info] unique_labels length:', len(unique_labels))

        rate_ = 1 - source_rate

        sigma2_b_expanded = sigma2_b.repeat(b, 1, 1, 1)
        mu_b_expanded = mu_b.repeat(b, 1, 1, 1)
        label_onehot = nn.functional.one_hot(
            inverse_indices, num_classes=len(unique_labels)
        ).float().to(x.device)

        mu = mu.view(b, c)
        sigma2 = sigma2.view(b, c)

        cluster_mus = torch.einsum('bc,bl->lc', mu, label_onehot) /( torch.einsum('bl->l', label_onehot).unsqueeze(1)+self.eps)
        weighted_sigma2 = torch.einsum('bc,bl->lc', sigma2, label_onehot) / (torch.einsum('bl->l', label_onehot).unsqueeze(1)+self.eps)
        mu_diff_squared = (mu.unsqueeze(1) - cluster_mus.unsqueeze(0))**2  # (batch_size, labelsize, channels)
        weighted_mu_diff_squared = torch.einsum('blc,bl->lc', mu_diff_squared, label_onehot) / (torch.einsum('bl->l', label_onehot).unsqueeze(1)+self.eps)
        cluster_sigma2s = weighted_sigma2 + weighted_mu_diff_squared
        
        cluster_mus_2 = torch.einsum("lc,bl->bc", cluster_mus, label_onehot).view(b, c, 1, 1)
        cluster_sigma2s_2 = torch.einsum("lc,bl->bc", cluster_sigma2s, label_onehot).view(b, c, 1, 1)

        x = (x - (rate_ * cluster_mus_2 + source_rate * mu_b_expanded)) * torch.rsqrt(
            (rate_ * cluster_sigma2s_2 + source_rate * sigma2_b_expanded) + self.eps
        )
        return x
    
    def forward(self, x):
        b, c, h, w = x.size()
        

        self.source_rate.data = torch.clamp(self.source_rate.data, 0, 1)
        mu_s = self._bn.running_mean.view(1, c, 1, 1)
        sigma2_s = self._bn.running_var.view(1, c, 1, 1)

        x_n = self.Lisc(x, mu_s, sigma2_s, self.source_rate)  # + self.eps
        
        weight = self._bn.weight.view(c, 1, 1)
        bias = self._bn.bias.view(c, 1, 1)
        x_n = x_n * weight + bias
        return x_n

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    

    def calculate_similarity(self, x, running_mean, running_var):
        # eps = 1e-30
        # x = x + eps
        b, c, h, w = x.size()
        source_mu = running_mean.reshape(1, c, 1, 1)
        source_sigma2 = running_var.reshape(1, c, 1, 1)

        sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True)
        sigma2_i, mu_i = torch.var_mean(x, dim=[2, 3], keepdim=True)

        # print("[info] source_mu.shape:", source_mu.shape)
        # print("[info] mu_b.shape:", mu_b.shape)

        # change the shape of source_mu and mu_b same as x
        mu_b = mu_b.repeat(b, 1, 1, 1)
        sigma2_b = sigma2_b.repeat(b, 1, 1, 1)
        source_mu = source_mu.repeat(b, 1, 1, 1)
        source_sigma2 = source_sigma2.repeat(b, 1, 1, 1)
        # print("[info] source_mu.shape:", source_mu.shape)
        # print("[info] mu_b.shape:", mu_b.shape)
        # dsi = source_mu - x
        # dti = mu_b - x
        # dst = source_mu - mu_b
        dsi = mu_i
        dti = mu_b - x
        dst = mu_b
        if torch.isnan(dsi).any():
            print("[Warning] NaNs found in dsi. Handling NaNs...")
        if torch.isnan(dst).any():
            print("[Warning] NaNs found in dst. Handling NaNs...")
        # if torch.isnan(dsi).any():
        # print("[Warning] NaNs found in similarity. Handling NaNs...")
        # print(f"[info]similarity:{x}")
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)
        # similarity = cos_similarity(dsi.view(b, c, -1), dst.view(b, c, -1))
        similarity = cos_similarity(x, mu_b)
        if torch.isnan(similarity).any():
            print("[Warning] NaNs found in similarity. Handling NaNs...")

        similarity = torch.clamp(similarity, -1.0, 1.0)
        curve_batch = torch.arccos(similarity)
        if torch.isnan(curve_batch).any():
            print("[Warning] NaNs found in curve_batch. Handling NaNs...")

        return similarity.cpu().detach().numpy(), curve_batch.cpu().detach().numpy()



def clust_rank(mat, use_ann_above_samples, initial_rank=None):
    s = mat.shape[0]

    if initial_rank is not None:
        orig_dist = torch.empty((1, 1), device=mat.device)
    elif s <= use_ann_above_samples:
        # Cosine distance
        norm_mat = torch.nn.functional.normalize(mat, p=2, dim=1)
        orig_dist = 1 - torch.mm(norm_mat, norm_mat.T)
        diag_indices = torch.arange(s, device=mat.device)
        orig_dist[diag_indices, diag_indices] = float('inf')
        initial_rank = torch.argmin(orig_dist, dim=1)

    # The Clustering Equation
    indices = torch.arange(s, device=mat.device)
    values = torch.ones(s, dtype=torch.float32, device=mat.device)
    A = torch.sparse_coo_tensor(
        torch.stack([indices, initial_rank]),
        values,
        size=(s, s)
    )
    A = A + torch.sparse_coo_tensor(
        torch.stack([indices, indices]),
        values,
        size=(s, s)
    )
    A = torch.sparse.mm(A, A.t())
    A = A.to_dense()
    A[indices, indices] = 0
    return A, orig_dist

def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        mask = (orig_dist * a) > min_sim
        a = torch.where(mask, torch.zeros_like(a), a)
    # Convert the dense matrix to a SciPy sparse matrix for connected components calculation
    a_scipy = sp.csr_matrix(a.cpu().numpy())
    num_clust, labels = sp.csgraph.connected_components(a_scipy, directed=True, connection='weak')
    labels = torch.tensor(labels, device=a.device)
    return labels, num_clust

def FINCH(data, initial_rank=None, use_ann_above_samples=70000):
    """ Simplified FINCH clustering algorithm for the first partition.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param use_ann_above_samples: Above this data size (number of samples) approximate nearest neighbors will be used to speed up neighbor
        discovery. For large scale data where exact distances are not feasible to compute, set this. [default = 70000]
    :return:
            c: Nx1 array. Cluster label for the first partition.
            num_clust: Number of clusters.
    """
    # Ensure data is a PyTorch tensor
    adj, orig_dist = clust_rank(data, use_ann_above_samples, initial_rank)
    group, num_clust = get_clust(adj, orig_dist)

    return group, num_clust