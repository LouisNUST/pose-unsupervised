# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pickle
from utils.transforms import get_affine_transform
import numpy as np
import itertools
import math

import torch
import torch.nn.functional as F
import torch.distributed as dist


def get_infonce_loss(embd1, embd2):
    """
    embd1: [N, C]
    emdb2: [N, C]
    """
    batch_size = embd1.shape[0]
    u_p = torch.sum(embd1 * embd2, dim=1).unsqueeze(1)  # [N, 1]
    u_n = torch.mm(embd1, embd2.t())  # [N, N]

    mask = torch.eye(batch_size).to(embd1.device)
    n_mask = 1 - mask
    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out examples

    pred_lgt = torch.cat([u_p, u_n], dim=1)
    pred_log = F.log_softmax(pred_lgt, dim=1)
    loss = -pred_log[:, 0].mean()
    return loss

def get_jsd_loss(embd1, embd2):
    """
    embd1: [N, C]
    emdb2: [N, C]
    """
    batch_size = embd1.shape[0]
    u = torch.mm(embd1, embd2.t())  # [N, N]

    mask = torch.eye(batch_size).to(embd1.device)
    n_mask = 1 - mask

    log_2 = math.log(2.)
    E_pos = log_2 - F.softplus(-u)
    E_neg = F.softplus(-u) + u - log_2

    E_pos = (E_pos * mask).sum() / mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos

    return loss

class JointsMSELoss(torch.nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss


class FundamentalLoss:
    def __init__(self, cfg):
        self.use_target_weight = cfg.LOSS.USE_TARGET_WEIGHT_FUND
        fundamental_matrix_path = os.path.join(cfg.DATASET.ROOT, 'testdata', 'fundamental_matrix.pkl')
        with open(fundamental_matrix_path, 'rb') as f:
            self.fundamental_matrix_dict = pickle.load(f)
        # move to current cuda device
        self.rank = dist.get_rank()
        self.device = torch.device('cuda', self.rank)
        for k, v in self.fundamental_matrix_dict.items():
            self.fundamental_matrix_dict[k] = torch.from_numpy(v).to(self.device, dtype=torch.float32)

    def __call__(self, joints_2d_list, target_weight, meta):
        """
        All inputs are from h36m.
        joints_2d_list: a list (4 views) of  [K, 16, 2], on gpu, already tran to img coordiantes
        target_weight: a list (4 views) of [K, 16, 1], on gpu
        meta: a list (4 views) of K dictionaries, cpu
        """
        assert isinstance(joints_2d_list[0], torch.Tensor)
        n_views = len(joints_2d_list)
        batch_size, n_joints = joints_2d_list[0].shape[:2]
        output_device = joints_2d_list[0].device
        subject = meta[0]['subject'].numpy()  # only 1 view
        assert batch_size == len(subject)

        homo_joints_2d_list = []
        for p in joints_2d_list:
            h_p = torch.cat((p, 
                torch.ones(batch_size, n_joints, 1).to(device=output_device)),
                dim=2)  # [k, 16, 3] homogeneous coordinates
            homo_joints_2d_list.append(h_p)

        # calculate fundamental loss
        pairs = list(itertools.permutations(list(range(n_views)), 2))  # 12 pairs
        loss = 0
        for idx, subj in enumerate(subject):
            for pair in pairs:
                F = self.fundamental_matrix_dict[(subj, pair[0], pair[1])]  # [3, 3]
                this_loss = torch.abs(torch.sum(torch.mm(homo_joints_2d_list[pair[1]][idx], F) * homo_joints_2d_list[pair[0]][idx], dim=1))  # [16]
                if self.use_target_weight:
                    this_loss *= torch.squeeze(target_weight[pair[1]][idx] * target_weight[pair[0]][idx])
                loss += this_loss.sum()
        loss /= (batch_size*len(pairs)*n_joints)
        return loss


class MILoss:
    """
      Warp discriminators and generate positive/negative pairs
    """
    def __init__(self, cfg, discriminator_dict):
        # wheter use_target_weight not added
        self.use_target_weight = cfg.LOSS.USE_TARGET_WEIGHT
        self.use_local_mi_loss = cfg.LOSS.USE_LOCAL_MI_LOSS
        self.use_global_mi_loss = cfg.LOSS.USE_GLOBAL_MI_LOSS
        self.measure = cfg.LOSS.MI_MEASURE
        self.neg_sample_per_pos = cfg.LOSS.MI_NEG_POS_RATIO
        self.positive_num = cfg.LOSS.MI_POSITIVE_NUM
        self.sigma = cfg.NETWORK.SIGMA
        # map 2d joints locations to heatmap
        self.feat = torch.from_numpy(cfg.NETWORK.IMAGE_SIZE / cfg.NETWORK.HEATMAP_SIZE)  # [2]
        if self.use_local_mi_loss:
            self.local_d = discriminator_dict['local_discriminator']
        if self.use_global_mi_loss:
            self.global_d = discriminator_dict['global_discriminator']

        if cfg.LOSS.SPECIFIC == 'org':
            self.extract_local_pairs = self.extract_local_pairs_org
        elif cfg.LOSS.SPECIFIC == 'one_image':
            self.extract_local_pairs = self.extract_local_pairs_one_img
        elif cfg.LOSS.SPECIFIC == 'joint':
            self.extract_local_pairs = self.extract_local_pairs_joints_specific
        else:
            assert 0, 'not implemented!'

    def _sample_indices(self, batch_size, n_locs, n_samples, replacement=True):
        """
        Return: indices of [batch_size, n_samples], each row ranges from 0 to n_locs
        """
        weights = torch.ones(batch_size, n_locs, dtype=torch.float)
        idx = torch.multinomial(weights, n_samples, replacement=replacement)
        return idx

    def sample_locations(self, enc, n_samples):
        '''Randomly samples locations from localized features.

        Used for saving memory.

        Args:
            enc: [N, L, C]
            n_samples: int 
        Returns:
            [N, n_samples, C]

        '''
        batch_size, n_locs = enc.shape[:2]
        idx = self._sample_indices(batch_size, n_locs, n_samples, replacement=True)  # [N, n_samples]
        adx = torch.arange(0, batch_size).long()
        enc = enc[adx[:, None], idx]
        return enc

    def extract_local_pairs_org(self, low_features, high_features, target_weight, meta):
        """
        Image specific now
        low_features: [N, 2048, 8, 8]
        high_features: [N, 256, 64, 64]
        target_weight: [N, 16, 1]
        Return: [N, K, 64, 64], [N, 256, 64, 64]
        """
        batch_size, c_high, h_high, w_high = high_features.size()
        _, c_low, h_low, w_low = low_features.size()
        assert h_high == w_high
        assert h_low == w_low
        if h_low == 8 and h_high == 64:
            factor = int(h_high / h_low)
            size, stride = 3, 1
            low_patches = low_features.unfold(2, size, stride).unfold(3, size, stride)  # [N, 2048, 6, 6, 3, 3]
            _, _, h_num, w_num, _, _ = low_patches.size()
            assert h_num == 6, 'number of patches {} not equal 6'.format(h_num)
            low_patches = low_patches.permute(0, 2, 3, 4, 5, 1).contiguous().view(batch_size, h_num*w_num, -1)  # [N, 6*6, 9*2048]

            # extract positive pairs
            pos_indices_high = self._sample_indices(batch_size, h_high, self.positive_num*2).view(batch_size, self.positive_num, 2)  # [N, K, 2]
            gt_locations = meta['joints_2d_transformed'] / self.feat + 0.5  # in order (w, h)
            gt_locations = gt_locations.to(dtype=torch.long).clamp_(min=0, max=h_high-1)
            pos_indices_high = torch.cat([pos_indices_high, gt_locations], dim=1)  # [N, K+16, 2]
            pos_indices_low = pos_indices_high / factor
            pos_indices_low = pos_indices_low.to(dtype=torch.long) - 1
            pos_indices_low.clamp_(min=0, max=h_num - 1)
            pos_indices_high = pos_indices_high[:, :, 1] * w_high + pos_indices_high[:, :, 0]  # [N, K+16]
            pos_indices_low = pos_indices_low[:, :, 1] * w_num + pos_indices_low[:, :, 0]  # [N, K+16]
            pos_batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None]

            high_patches = high_features.permute(0, 2, 3, 1).view(batch_size, -1, c_high)  # [N, 64*64, 256], take care that img_h is in front of img_w
            high_pos = high_patches[pos_batch_indices, pos_indices_high].transpose(1, 2)  # [N, 256, K+16]
            low_pos = low_patches[pos_batch_indices, pos_indices_low].transpose(1, 2)  # [N, 9*2048, K+16]

            # extract negative pairs
            high_neg = high_pos.unsqueeze(2).expand(-1, -1, self.neg_sample_per_pos, -1).contiguous().view(batch_size, c_high, -1) # [N, 256, Q*(K+16)]
            def _neg_batch_indices(n):
                x = [[i for i in range(n) if i != k] for k in range(n)]
                return torch.tensor(list(itertools.chain(*x)), dtype=torch.long)
            neg_batch_indices = _neg_batch_indices(batch_size)  # [N*(N-1)]
            low_neg = low_patches[neg_batch_indices, ...].view(batch_size, batch_size-1, low_patches.size(1), -1)  # [N, N-1, 6*6, 9*2048]
            low_neg = low_neg.view(batch_size, -1, low_patches.size(2))  # [N, (N-1)*6*6, 9*2048]
            low_neg = self.sample_locations(low_neg, high_neg.size(-1))  # [N, Q*(K+16), 9*2048]

            # convert to [N, C, L]
            low_neg = low_neg.transpose(1, 2)  # [N, 9*2048, Q*(K+16+more)], [N, C, L]

            assert low_neg.size(2) == high_neg.size(2)
        else:
            assert 0, 'not implemented feature map size, low:{}, high:{}'.format(low_features.size(2),
                high_features.size(2))

        return low_pos, high_pos, low_neg, high_neg

    def extract_local_pairs_one_img(self, low_features, high_features, target_weight, meta):
        """
        Image specific now
        low_features: [N, 2048, 8, 8]
        high_features: [N, 256, 64, 64]
        target_weight: [N, 16, 1]
        Return: [N, K, 64, 64], [N, 256, 64, 64]

        no unfold now
        """
        batch_size, c_high, h_high, w_high = high_features.size()
        _, c_low, h_low, w_low = low_features.size()
        assert h_high == w_high
        assert h_low == w_low
        if h_low == 8 and h_high == 64:
            factor = int(h_high / h_low)
            size, stride = 3, 1
            low_patches = low_features.unfold(2, size, stride).unfold(3, size, stride)  # [N, 2048, 6, 6, 3, 3]
            _, _, h_num, w_num, _, _ = low_patches.size()
            assert h_num == 6, 'number of patches {} not equal 6'.format(h_num)
            low_patches = low_patches.permute(0, 2, 3, 4, 5, 1).contiguous().view(batch_size, h_num*w_num, -1)  # [N, 6*6, 9*2048]

            # extract positive pairs
            pos_indices_high = self._sample_indices(batch_size, h_high, self.positive_num*2).view(batch_size, self.positive_num, 2)  # [N, K, 2]
            gt_locations = meta['joints_2d_transformed'] / self.feat + 0.5  # in order (w, h)
            gt_locations = gt_locations.to(dtype=torch.long).clamp_(min=0, max=h_high-1)
            pos_indices_high = torch.cat([pos_indices_high, gt_locations], dim=1)  # [N, K+16, 2]
            pos_indices_low = pos_indices_high / factor
            pos_indices_low = pos_indices_low.to(dtype=torch.long) - 1
            pos_indices_low.clamp_(min=0, max=h_num - 1)
            pos_indices_high = pos_indices_high[:, :, 1] * w_high + pos_indices_high[:, :, 0]  # [N, K+16]
            pos_indices_low = pos_indices_low[:, :, 1] * w_num + pos_indices_low[:, :, 0]  # [N, K+16]
            pos_batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None]

            high_patches = high_features.permute(0, 2, 3, 1).view(batch_size, -1, c_high)  # [N, 64*64, 256], take care that img_h is in front of img_w
            high_pos = high_patches[pos_batch_indices, pos_indices_high].transpose(1, 2)  # [N, 256, K+16]
            low_pos = low_patches[pos_batch_indices, pos_indices_low].transpose(1, 2)  # [N, 9*2048, K+16]

            # extract negative pairs
            num_anchor = high_pos.size(-1)
            high_neg = high_pos.unsqueeze(2).expand(-1, -1, self.neg_sample_per_pos, -1).contiguous().view(batch_size, c_high, -1) # [N, 256, Q(K+16)]
            all_indices_low = torch.arange(low_patches.size(1))  # 36
            weights = torch.ne(all_indices_low[None, :], pos_indices_low.view(-1)[:, None]).to(dtype=torch.float)  # [N(K+16), 36]
            idx = torch.multinomial(weights, self.neg_sample_per_pos, replacement=True)  # [N(K+16), Q]
            low_neg_indices = idx.view(batch_size, num_anchor, self.neg_sample_per_pos).transpose(1, 2).contiguous().view(batch_size, -1)  # [N, Q(K+16)]
            low_neg = low_patches[pos_batch_indices, low_neg_indices]  # [N, Q(K+16), 9*2048]

            # convert to [N, C, L]
            low_neg = low_neg.transpose(1, 2)  # [N, 9*2048, Q*(K+16)], [N, C, L]

            assert low_neg.size(2) == high_neg.size(2)
        else:
            assert 0, 'not implemented feature map size, low:{}, high:{}'.format(low_features.size(2),
                high_features.size(2))

        return low_pos, high_pos, low_neg, high_neg

    def _sample_far_indices(self, loc, max_len=64):
        """
        loc : [N, 16]
        return : [N, 16, 64*64]  -> [N, 16, Q]
        """
        batch_size, num_joints = loc.shape
        radius = self.sigma * 3
        idx = torch.arange(-radius, radius+1)
        grid = idx[:, None] * max_len + idx[None, :]  # [2r+1, 2r+1]
        grid = grid.view(-1)  # [(2r+1)*(2r+1)]
        masked_loc = loc.view(-1)[:, None] + grid[None, :]  # [N*16, (2r+1)*(2r+1)]
        masked_loc.clamp_(min=0, max=max_len*max_len-1)
        masked_loc = masked_loc.view(batch_size, num_joints, -1)  # [N, 16, (2r+1)*(2r+1)]
        masked_loc += torch.arange(batch_size).view(batch_size, 1, 1) * max_len * max_len
        masked_loc = masked_loc.view(batch_size*num_joints, -1)  # [N*16, (2r+1)*(2r+1)]
        weights = torch.ones(masked_loc.shape[0], batch_size * max_len * max_len)  # [N*16, N*64*64]
        batch_indices = torch.arange(weights.shape[0], dtype=torch.long)[:, None]
        weights[batch_indices, masked_loc] = 0
        weights = weights.view(batch_size, num_joints, -1)   # [N, 16, N*64*64]
        weight = weights[0]
        for w in weights:
            weight *= w
        weights = weight[None, ...].expand(batch_size, -1, -1).contiguous().view(batch_size*num_joints, -1)  # [N*16, N*64*64]
        indices = torch.multinomial(weights, self.neg_sample_per_pos, replacement=True).view(-1)  # [N*16*Q]
        return indices

    def extract_local_pairs_joints_specific(self, low_features, high_features, target_weight, meta):
        batch_size, c_high, h_high, w_high = high_features.size()
        _, c_low, h_low, w_low = low_features.size()
        assert h_high == w_high
        assert h_low == w_low
        if h_low == 8 and h_high == 64:
            assert 0, 'not implemented feature map size, low:{}, high:{}'.format(low_features.size(2),
                high_features.size(2))
        elif h_low == 64 and h_high == 64:
            feature_patches = low_features.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, c_low)  # [N, 64*64, 256]
            batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None]  # [N, 1]

            # extract gt pos pairs
            gt_locations = meta['joints_2d_transformed'] / self.feat + 0.5  # in order (w, h)
            gt_locations = gt_locations.to(dtype=torch.long).clamp_(min=0, max=h_high-1)  # [N, 16, 2]
            gt_locations = gt_locations[:, :, 1] * w_low + gt_locations[:, :, 0]  # [N, 16]
            gt_candidates = feature_patches[batch_indices, gt_locations, :]  # [N, 16, 256]
            num_joints = gt_locations.size(-1)

            joints_vis = meta['joints_vis'][:, :, 0]  # [N, 16]
            joints_vis_tran = joints_vis.transpose(0, 1)  # [16, N]
            pos_pair_choice_map = joints_vis_tran[:, None, :] * joints_vis_tran[:, :, None]  # [16, N, N]
            pos_pair_choice_map = pos_pair_choice_map.to(dtype=torch.float)
            mask = 1 - torch.eye(gt_candidates.size(0), dtype=torch.float)[None, :, :]
            pos_pair_choice_map *= mask
            none_zero_indices = pos_pair_choice_map.nonzero()  # [K, 3]
            low_pos_gt = gt_candidates[none_zero_indices[:, 1], none_zero_indices[:, 0], :]  # [K, 256]
            high_pos_gt = gt_candidates[none_zero_indices[:, 2], none_zero_indices[:, 0], :]  # [K, 256]
            # extract gt neg pairs
            visible_joints_indices = joints_vis.view(-1).nonzero().view(-1)  # [R]
            low_neg_gt_indices = self._sample_far_indices(gt_locations, max_len=h_high)  # [N*16*Q]
            low_neg_gt_indices = low_neg_gt_indices.view(batch_size*num_joints,-1)   # [N*16, Q]
            low_neg_gt_indices = low_neg_gt_indices[visible_joints_indices, ...].view(-1)  # [R*Q]
            low_neg_gt = feature_patches.view(-1, c_low)[low_neg_gt_indices, ...]  # [R*Q, 256]
            high_neg_gt = gt_candidates.view(-1, c_low)[visible_joints_indices, ...]  # [R, 256]
            high_neg_gt = high_neg_gt[:, None, :]
            high_neg_gt = high_neg_gt.expand(-1, self.neg_sample_per_pos, -1).contiguous().view(-1, c_low)  # [R*Q, 256]

            # extract bg gt pairs, sample many pairs, not a complete graph
            weights = torch.ones((batch_size*h_low*h_low, ))  # [N*64*64]
            flatten_gt_locations = gt_locations + torch.arange(batch_size)[:, None] * h_low * h_low
            flatten_gt_locations = flatten_gt_locations.view(-1)  # [N*16]
            weights[flatten_gt_locations] = 0
            low_bg_indices = torch.multinomial(weights, self.positive_num*2, replacement=True)  # [T*2]
            low_pos_bg = feature_patches.view(-1, c_low)[low_bg_indices[:self.positive_num]]  # [T, 256]
            high_pos_bg = feature_patches.view(-1, c_low)[low_bg_indices[self.positive_num:]]  # [T, 256]

            # final pos, final neg
            low_pos = torch.cat([low_pos_gt, low_pos_bg], dim=0)  # [K+T, 256]
            high_pos = torch.cat([high_pos_gt, high_pos_bg], dim=0)  # [K+T, 256]
            low_neg = low_neg_gt  # [R*Q, 256]
            high_neg = high_neg_gt  # [R*Q, 256]

            low_pos = low_pos.transpose(0, 1)
            high_pos = high_pos.transpose(0, 1)  # [256, K+T]
            low_neg = low_neg.transpose(0, 1)
            high_neg = high_neg.transpose(0, 1)  # [256, R*Q]
        else:
            assert 0, 'not implemented feature map size, low:{}, high:{}'.format(low_features.size(2),
                high_features.size(2))
        return low_pos, high_pos, low_neg, high_neg

    def extract_global_positive_pairs(self, low_features, high_features, target_weight, meta):
        assert 0, 'not implemented method'
        return

    def extract_global_negative_pairs(self, low_features, high_features, target_weight, meta):
        assert 0, 'not implemented method'
        return

    def get_positive_expectation(self, p_samples, measure, average=True):
        """Computes the positive part of a divergence / difference.

        Args:
            p_samples: Positive samples.
            measure: Measure to compute for.
            average: Average the result over samples.

        Returns:
            torch.Tensor

        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'DV':
            Ep = p_samples
        elif measure == 'H2':
            Ep = 1. - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            raise_measure_error(measure)

        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, measure, average=True):
        """Computes the negative part of a divergence / difference.

        Args:
            q_samples: Negative samples.
            measure: Measure to compute for.
            average: Average the result over samples.

        Returns:
            torch.Tensor

        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples - 1.)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'DV':
            Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            raise_measure_error(measure)

        if average:
            return Eq.mean()
        else:
            return Eq

    def get_infonce_loss(self, pos_scores, neg_scores):
        """
        pos_scores: [N, K+16]
        neg_scores: [N, Q(K+16)]
        """
        batch_size, npos = pos_scores.shape
        assert neg_scores.shape[1] == self.neg_sample_per_pos * npos
        pos_scores = pos_scores[:, None, :]  # [N, 1, K+16]
        neg_scores = neg_scores.view(batch_size, self.neg_sample_per_pos, npos)  # [N, Q, K+16]
        scores = torch.cat([pos_scores, neg_scores], dim=1)  # [N, 1+Q, K+16]
        return -F.log_softmax(scores, dim=1)[:, 0, :].mean()

    def contrastive_gradient_penalty(self, network, input, penalty_amount=1.):
        """Contrastive gradient penalty.

        This is essentially the loss introduced by Mescheder et al 2018.

        Args:
            network: Network to apply penalty through.
            input: Input or list of inputs for network.
            penalty_amount: Amount of penalty.

        Returns:
            torch.Tensor: gradient penalty loss.

        """
        def _get_gradient(inp, output):
            gradient = torch.autograd.grad(outputs=output, inputs=inp,
                                     grad_outputs=torch.ones_like(output),
                                     create_graph=True, retain_graph=True,
                                     only_inputs=True, allow_unused=True)[0]
            return gradient

        if not isinstance(input, (list, tuple)):
            input = [input]

        input = [inp.detach() for inp in input]
        input = [inp.requires_grad_() for inp in input]

        with torch.set_grad_enabled(True):
            # output = network(*input)[-1]
            output = network(*input)
        gradient = _get_gradient(input, output)
        gradient = gradient.view(gradient.size()[0], -1)
        penalty = (gradient ** 2).sum(1).mean()

        return penalty * penalty_amount


    def __call__(self, low_features, high_features, target_weight, meta):
        """
           Only one view is passed in.
           Discriminator takes [N, C, H, W] or [N, C, L] as input
        low_features: [N, 2048, 8, 8], on gpu
        high_features: [N, 256, 64, 64], on gpu
        target_weight: [N, 16, 1], on gpu
        meta: dictionary of N elements, on cpu
        """
        local_loss = 0
        global_loss = 0
        if self.use_local_mi_loss:
            local_pos_low_features, local_pos_high_features, local_neg_low_features, local_neg_high_features = self.extract_local_pairs(
                low_features, high_features, target_weight, meta)
            local_pos_scores = self.local_d(local_pos_low_features, local_pos_high_features)  # [N, H, W] or [N, L]
            local_neg_scores = self.local_d(local_neg_low_features, local_neg_high_features)
            gp_loss_pos = self.contrastive_gradient_penalty(self.local_d, [local_pos_low_features, local_pos_high_features], penalty_amount=1.)
            gp_loss_neg = self.contrastive_gradient_penalty(self.local_d, [local_neg_low_features, local_neg_high_features], penalty_amount=1.)
            gp_loss = 0.5 * (gp_loss_pos + gp_loss_neg)
            if self.measure == 'NCE':
                local_loss = self.get_infonce_loss(local_pos_scores, local_neg_scores)
            else:
                Epos = self.get_positive_expectation(local_pos_scores, measure=self.measure)  # scalar
                Eneg = self.get_negative_expectation(local_neg_scores, measure=self.measure)
                local_loss = Eneg - Epos
            local_loss += gp_loss

        if self.use_global_mi_loss:
            global_pos_low_features, global_pos_high_features = self.extract_global_positive_pairs(low_features, high_features, target_weight, meta)
            global_neg_low_features, global_neg_high_features = self.extract_global_negative_pairs(low_features, high_features, target_weight, meta)
            global_pos_scores = self.global_d(global_pos_low_features, global_pos_high_features)  # [N, H, W] or [N, L]
            global_neg_scores = self.global_d(global_neg_low_features, global_neg_high_features)
            Epos = self.get_positive_expectation(global_pos_scores, measure=self.measure)  # scalar
            Eneg = self.get_negative_expectation(global_neg_scores, measure=self.measure)
            global_loss = Eneg - Epos

        return global_loss, local_loss


class ViewMILoss:
    def __init__(self, cfg, discriminator_dict):
        self.view_d = discriminator_dict['view_discriminator']
        self.measure = cfg.LOSS.VIEW_MI_MEASURE
        self.view1_num = cfg.VIEW_DISCRIMINATOR.VIEW_ONE_NUM
        self.view2_num = cfg.VIEW_DISCRIMINATOR.VIEW_TWO_NUM
        assert self.view1_num <= self.view2_num
    
    def __call__(self, joints_2d_list):
        """
        Heatmaps_list: a 4-view list of heatmaps [N, 16, 64, 64]
        """
        if self.view1_num == 1:
            joints_2d_view1 = joints_2d_list[0]
            joints_2d_view2 = torch.cat(joints_2d_list[1:], dim=1)
        elif self.view1_num == 3:
            joints_2d_view1 = torch.cat(joints_2d_list[:3], dim=1)
            joints_2d_view2 = joints_2d_list[-1]
        else:
            joints_2d_view1 = torch.cat(joints_2d_list[:self.view1_num], dim=1)
            joints_2d_view2 = torch.cat(joints_2d_list[self.view1_num:], dim=1)
        embd_view1, embd_view2 = self.view_d(joints_2d_view1, joints_2d_view2)  # [N, C]

        if self.measure == 'NCE':
            loss = get_infonce_loss(embd_view1, embd_view2)
        elif self.measure == 'JSD':
            loss = get_jsd_loss(embd_view1, embd_view2)
        else:
            assert 0, 'not implemented yet'

        return loss


class JointsMILoss:
    def __init__(self, cfg, discriminator_dict):
        self.joints_d = discriminator_dict['joints_discriminator']
        self.measure = cfg.LOSS.JOINTS_MI_MEASURE
        self.var1_num = cfg.JOINTS_DISCRIMINATOR.VAR_ONE_NUM
        self.var2_num = cfg.JOINTS_DISCRIMINATOR.VAR_TWO_NUM
        self.joints_num = cfg.NETWORK.NUM_JOINTS
        assert self.var1_num <= self.var2_num
        assert self.var1_num + self.var2_num == self.joints_num
        self.var1_idx = cfg.JOINTS_DISCRIMINATOR.VAR_ONE_IDX
        self.var2_idx = np.array(list(set(range(self.joints_num)) - set(self.var1_idx)), dtype=np.int)
        assert len(self.var1_idx) == self.var1_num

        # tranf idx to gpu
        rank = dist.get_rank()
        device = torch.device('cuda', rank)
        self.var1_idx = torch.from_numpy(self.var1_idx).to(device=device)
        self.var2_idx = torch.from_numpy(self.var2_idx).to(device=device)
    
    def __call__(self, joints_2d, var2_no_grad=False):
        """
        joints2d: [N, 16, 2]
        """
        joints_2d_var1 = joints_2d[:, self.var1_idx, :]
        joints_2d_var2 = joints_2d[:, self.var2_idx, :]
        if var2_no_grad:
            joints_2d_var2 = joints_2d_var2.detach()
        embd_var1, embd_var2 = self.joints_d(joints_2d_var1, joints_2d_var2)  # [N, C]

        if self.measure == 'NCE':
            loss = get_infonce_loss(embd_var1, embd_var2)
        elif self.measure == 'JSD':
            loss = get_jsd_loss(embd_var1, embd_var2)
        else:
            assert 0, 'not implemented yet'

        return loss


class HeatmapMILoss:
    def __init__(self, cfg, discriminator_dict):
        # wheter use_target_weight not added
        self.use_target_weight = cfg.LOSS.USE_TARGET_WEIGHT
        self.measure = cfg.LOSS.HEATMAP_MI_MEASURE
        self.sigma = cfg.NETWORK.SIGMA
        # map 2d joints locations to heatmap
        self.feat = torch.from_numpy(cfg.NETWORK.IMAGE_SIZE / cfg.NETWORK.HEATMAP_SIZE)  # [2]
        self.heatmap_d = discriminator_dict['heatmap_discriminator']

    def _sample_some_indices(self, loc, max_len=64):
        """
        loc : [N]
        return : [N, 64*64]  -> [N, Q]
        """
        batch_size = loc.shape[0]
        radius = self.sigma * 3 + 2
        idx = torch.arange(-radius, radius+1)
        grid = idx[:, None] * max_len + idx[None, :]  # [2r+1, 2r+1]
        grid = grid.view(-1)  # [(2r+1)*(2r+1)]
        masked_loc = loc.view(-1)[:, None] + grid[None, :]  # [N, (2r+1)*(2r+1)]
        masked_loc.clamp_(min=0, max=max_len*max_len-1)

        # sample in radius R, to save GPU memory
        gt_weights = torch.ones((batch_size, masked_loc.shape[1]), dtype=torch.float)
        sampled_indices = torch.multinomial(gt_weights, int(masked_loc.shape[1]/2), replacement=False)  # [N, (2r+1)*(2r+1)/2]
        batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None]  # [N, 1]
        high_response_indices = masked_loc[batch_indices, sampled_indices]  # [N, (2r+1)*(2r+1)/2]

        # sample from other parts than radius R
        neg_weights = torch.ones((batch_size, max_len*max_len), dtype=torch.float)  # [N, 64*64]
        neg_weights[torch.arange(batch_size)[:, None], masked_loc] = 0

        low_response_indices = torch.multinomial(neg_weights, int(masked_loc.size(1)/4), replacement=False)  # [N, (2r+1)*(2r+1)/4]
        indices = torch.cat([high_response_indices, low_response_indices], dim=1)  # [N, K]

        return indices

    def extract_heatmap_pairs(self, low_features, high_features, target_weight, meta, joint_idx):
        """
          Heatmap prob (X) scalar 0-1 <--> Image feature (Y) 256d
        low_features: [N, 256, 64, 64]
        heatmap: [N, 16, 64, 64]
        target_weight: [N, 16, 1]
        Return: [N*Q*Q, 256+1]
        """
        batch_size, c_high, h_high, w_high = high_features.size()
        _, c_low, h_low, w_low = low_features.size()
        assert h_high == w_high
        assert h_low == w_low
        if h_low == 64 and h_high == 64:
            feature_patches = low_features.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, c_low)  # [N, 64*64, 256]
            heatmap = high_features[:, joint_idx, :, :].view(batch_size, -1)  # [N, 64*64]
            batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None]  # [N, 1]

            # reconstruct gt indices
            gt_locations = meta['joints_2d_transformed'] / self.feat + 0.5  # in order (w, h)
            gt_locations = gt_locations.to(dtype=torch.long).clamp_(min=0, max=h_high-1)  # [N, 16, 2]
            gt_locations = gt_locations[:, :, 1] * w_low + gt_locations[:, :, 0]  # [N, 16]
            gt_locations = gt_locations[:, joint_idx]  # [N]
            joints_vis = meta['joints_vis'][:, joint_idx, 0]  # [N]
            visible_gt_indices = joints_vis.nonzero().squeeze(1)  # [Z]
            random_sampled_gt_indices = torch.randint(0, h_low*h_high, size=(batch_size-visible_gt_indices.shape[0],), dtype=torch.long)
            new_gt_locations = torch.cat([gt_locations[visible_gt_indices], random_sampled_gt_indices])  # [N]

            # sample according to gt locations, radius 7*7 0.1~1 plus other 6*6 zero elements
            # [N, 7*7+(7*7)//4]
            indices = self._sample_some_indices(new_gt_locations, max_len=h_high)

            sampled_low = feature_patches[batch_indices, indices, :]  # [N, 7*7+(7*7)//4, 256]
            sampled_heatmap = heatmap[batch_indices, indices]  # [N, 7*7+(7*7)//4]

            sampled_low = sampled_low[:, :, None, :]  # [N, 7*7+(7*7)//4, 1, 256]
            sampled_heatmap = sampled_heatmap[:, None, :, None]  # [N, 1, 7*7+(7*7)//4, 1]
            sampled_low = sampled_low.expand(-1, -1, indices.shape[1], -1)
            sampled_heatmap = sampled_heatmap.expand(-1, indices.shape[1], -1, -1)
            all_pairs = torch.cat([sampled_heatmap, sampled_low], dim=-1)  # [N, 7*7+(7*7)//4, 7*7+(7*7)//4, 1+256]
            all_pairs = all_pairs.view(-1, all_pairs.shape[-1])  # [N*Q*Q, C]
        else:
            assert 0, 'not implemented feature map size, low:{}, high:{}'.format(low_features.size(2),
                high_features.size(2))
        return all_pairs

    def get_infonce_loss(self, u):
        """
        u [N, Q*Q]
        """
        batch_size, area = u.shape
        edge = int(math.sqrt(area))
        u = u.view(batch_size, edge, edge)  # [N, Q, Q]

        p_indices = torch.arange(edge)
        u_p = u[:, p_indices, p_indices]  # [N, Q]
        mask = torch.eye(edge).to(u.device)[None, ...]  # [1, Q, Q]
        n_mask = 1 - mask
        u_n = (n_mask * u) - (10. * mask)  # [N, Q, Q]

        pred_lgt = torch.cat([u_p[..., None], u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)  # [N, Q, Q+1]
        loss = -pred_log[:, :, 0].mean()
        return loss

    def get_jsd_loss(self, u):
        """
        u [N, Q*Q]
        """
        batch_size, area = u.shape
        edge = int(math.sqrt(area))
        u = u.view(batch_size, edge, edge)  # [N, Q, Q]

        mask = torch.eye(edge).to(u.device)[None, ...]  # [1, Q, Q]
        n_mask = 1 - mask

        log_2 = math.log(2.)
        E_pos = log_2 - F.softplus(-u)
        E_neg = F.softplus(-u) + u - log_2

        E_pos = (E_pos * mask).sum() / (mask.sum() * batch_size)
        E_neg = (E_neg * n_mask).sum() / (n_mask.sum() * batch_size)
        loss = E_neg - E_pos

        return loss

    def __call__(self, low_features, high_features, target_weight, meta, joint_idx):
        """
           Only one view is passed in.
           Discriminator takes [N, C] as input
        low_features: [N, 256, 64, 64], on gpu
        high_features: [N, 16, 64, 64], on gpu
        target_weight: [N, 16, 1], on gpu
        meta: dictionary of N elements, on cpu
        """
        batch_size = low_features.shape[0]
        heatmap_loss = 0
        all_pairs = self.extract_heatmap_pairs(
            low_features, high_features, target_weight, meta, joint_idx)  # [N*Q*Q, C]

        scores = self.heatmap_d(all_pairs)  # [N*Q*Q, 1]
        scores = scores.view(batch_size, -1)

        if self.measure == 'NCE':
            heatmap_loss = self.get_infonce_loss(scores)
        elif self.measure == 'JSD':
            heatmap_loss = self.get_jsd_loss(scores)

        return heatmap_loss