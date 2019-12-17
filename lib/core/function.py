# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import h5py
import numpy as np

import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back_th, generate_integral_preds_2d_th, transform_back_th
from utils.vis import save_debug_images
import torch.distributed as dist

from utils.gradients import check_grad_norm
from collections import OrderedDict

logger = logging.getLogger(__name__)


def fuse_routing(raw_features, aggre_features, is_aggre, meta):
    if not is_aggre:
        return raw_features

    output = []
    for r, a, m in zip(raw_features, aggre_features, meta):
        view = torch.zeros_like(a)
        batch_size = a.size(0)
        for i in range(batch_size):
            s = m['source'][i]
            view[i] = 3/5 * a[i] + 2/5 * r[i] if s == 'h36m' else r[i]
        output.append(view)
    return output

def select_out_h36m(raw_features, aggre_features, meta):
    """
    raw_features: [tensor*4]
    aggre_features: [tensor*4]
    Return: raw_features from h36m, agg_features from h36m, 
    """
    raw_h36m = []
    agg_h36m = []
    for r, a, m in zip(raw_features, aggre_features, meta):
        indices = torch.tensor([source == 'h36m' for source in m['source']], dtype=torch.uint8)
        raw_h36m.append(r[indices])
        agg_h36m.append(a[indices])
    return raw_h36m, agg_h36m


def select_out_h36m_meta(meta, key_list):
    """
    Seletc out useful h36m meta items and keys from original meta
    """
    new_meta = []
    for idx, m in enumerate(meta):
        indices = torch.tensor([source == 'h36m' for source in m['source']], dtype=torch.uint8)
        if indices.sum() == 0:
            break
        new_meta.append({})
        for key in key_list:
            new_meta[idx][key] = m[key][indices]
    assert len(new_meta) == 0 or len(new_meta) == len(meta)
    return new_meta


def percent_of_datasource(meta):
    view_0_meta = meta[0]
    batch_num = len(view_0_meta['source'])
    res_dict = {}
    for source in view_0_meta['source']:
        num = res_dict.setdefault(source, 0)
        res_dict[source] += 1
    res_string = ''
    for k, v in res_dict.items():
        res_string += '{} {:.1%}\t'.format(k, v/batch_num)
    return res_string


def train(config, data, model_dict, criterion_dict, optim_dict, epoch, output_dir,
          writer_dict, rank):
    device = torch.device('cuda', rank)
    is_aggre = config.NETWORK.AGGRE
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if rank == 0:
        losses = AverageMeter()
        avg_acc = AverageMeter()

        mse_losses = AverageMeter()

        if config.LOSS.USE_CONSISTENT_LOSS:
            consistent_losses = AverageMeter()

        if config.LOSS.USE_FUNDAMENTAL_LOSS:
            fund_losses = AverageMeter()

        if config.LOSS.USE_GLOBAL_MI_LOSS:
            global_mi_losses = AverageMeter()
        if config.LOSS.USE_LOCAL_MI_LOSS:
            local_mi_losses = AverageMeter()

        if config.LOSS.USE_DOMAIN_TRANSFER_LOSS:
            domain_losses_g = AverageMeter()
            domain_losses_d = AverageMeter()
            domain_accuracy_d = AverageMeter()

        if config.LOSS.USE_VIEW_MI_LOSS:
            view_losses_g = AverageMeter()
            view_losses_d = AverageMeter()

        if config.LOSS.USE_JOINTS_MI_LOSS:
            jmi_losses_g = AverageMeter()
            jmi_losses_d = AverageMeter()

        if config.LOSS.USE_HEATMAP_MI_LOSS:
            hmi_losses_g = AverageMeter()
            hmi_losses_d = AverageMeter()

        if config.LOSS.WATCH_GRAD_NORM:
            mse_grad_norms = AverageMeter()
            if config.LOSS.USE_FUNDAMENTAL_LOSS:
                fund_grad_norms = AverageMeter()
            if config.LOSS.USE_JOINTS_MI_LOSS:
                jmi_grad_norms = AverageMeter()
            if config.LOSS.USE_HEATMAP_MI_LOSS:
                hmi_grad_norms = AverageMeter()
            if config.LOSS.USE_VIEW_MI_LOSS:
                vmi_grad_norms = AverageMeter()

    # switch to train mode
    for model in model_dict.values():
        model.train()

    end = time.time()
    for i, (input, target, weight, meta) in enumerate(data):
        """
        input: a list(4 views) of [N, 3, H, W]
        target: a list of [N, 16, h, w]
        weight: a list of [N, 16, 1]
        meta: a list of dictionaries
        """
        data_time.update(time.time() - end)
        input = [view.to(device, non_blocking=False) for view in input]

        raw_features, aggre_features, low_features, high_features = model_dict['base_model'](input)

        if is_aggre and config.TEST.FUSE_OUTPUT:
            output = fuse_routing(raw_features, aggre_features, is_aggre, meta)
        else:
            output = raw_features

        loss = 0
        mse_loss = 0
        consistent_loss = 0
        fund_loss = 0
        global_mi_loss = 0
        local_mi_loss = 0
        total_norm_local = 0
        target_cuda = []
        weight_cuda = []

        # mse loss on single view, with ground truth heat maps
        for t, w, r in zip(target, weight, raw_features):
            t = t.to(device, non_blocking=False)
            w = w.to(device, non_blocking=False)
            target_cuda.append(t)
            weight_cuda.append(w)
            mse_loss += criterion_dict['mse_weights'](r, t, w) * config.LOSS.MSE_LOSS_WEIGHT
        loss += mse_loss

        # mse loss on agg output, with pseudo heat maps
        if is_aggre:
            for t, w, o in zip(target_cuda, weight_cuda, output):
                mse_loss += criterion_dict['mse_weights'](o, t, w) * config.LOSS.MSE_LOSS_WEIGHT
            loss += mse_loss

        # mutual infomation loss
        if config.LOSS.USE_GLOBAL_MI_LOSS or config.LOSS.USE_LOCAL_MI_LOSS:
            # ########### update discriminator
            d_loss = 0
            for l, h, w, m in zip(high_features, high_features, weight_cuda, meta):
                global_loss_detach, local_loss_detach = criterion_dict['mutual_info'](l.detach(), h.detach(), w, m)
                if config.LOSS.USE_GLOBAL_MI_LOSS:
                    d_loss += global_loss_detach * config.LOSS.GLOBAL_MI_LOSS_WEIGHT
                if config.LOSS.USE_LOCAL_MI_LOSS:
                    d_loss += local_loss_detach * config.LOSS.LOCAL_MI_LOSS_WEIGHT

            if config.LOSS.USE_GLOBAL_MI_LOSS:
                optim_dict['global_discriminator'].zero_grad()
            if config.LOSS.USE_LOCAL_MI_LOSS:
                optim_dict['local_discriminator'].zero_grad()
            d_loss.backward()

            # clip gradient of discriminator
            if config.LOSS.USE_GRADIENT_CLIP:
                if config.LOSS.USE_LOCAL_MI_LOSS:
                    total_norm_local = torch.nn.utils.clip_grad_norm_(model_dict['local_discriminator'].parameters(), max_norm=1)
                if config.LOSS.USE_GLOBAL_MI_LOSS:
                    total_norm_global = torch.nn.utils.clip_grad_norm_(model_dict['global_discriminator'].parameters(), max_norm=1)

            if config.LOSS.USE_GLOBAL_MI_LOSS:
                optim_dict['global_discriminator'].step()
            if config.LOSS.USE_LOCAL_MI_LOSS:
                optim_dict['local_discriminator'].step()

            # ########### update backbone
            for l, h, w, m in zip(high_features, high_features, weight_cuda, meta):
                global_loss, local_loss = criterion_dict['mutual_info'](l, h, w, m)
                if config.LOSS.USE_GLOBAL_MI_LOSS:
                    loss += global_loss * config.LOSS.GLOBAL_MI_LOSS_WEIGHT
                    global_mi_loss += global_loss.item() * config.LOSS.GLOBAL_MI_LOSS_WEIGHT
                if config.LOSS.USE_LOCAL_MI_LOSS:
                    loss += local_loss * config.LOSS.LOCAL_MI_LOSS_WEIGHT
                    local_mi_loss += local_loss.item() * config.LOSS.LOCAL_MI_LOSS_WEIGHT

        # domain transfer loss
        if config.LOSS.USE_DOMAIN_TRANSFER_LOSS:
            # h36m label -> 0
            # mpii label -> 1
            # update discriminator
            domain_loss_d = 0
            domain_acc_d = 0
            label = torch.tensor([source != 'h36m' for source in meta[0]['source']], device=device, dtype=torch.float)
            label_d = label - label * 0.1
            label_d = label + (1 - label) * 0.1  # (0.1, 0.9)
            for l in low_features:
                d_score = model_dict['domain_discriminator'](l.detach()).squeeze()  # [4*N]
                domain_loss_d += criterion_dict['bce'](d_score, label_d)
                domain_acc_d += torch.eq(torch.ge(d_score, 0.5), label)
            optim_dict['domain_discriminator'].zero_grad()
            domain_loss_d.backward()
            # for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            #     print(p.grad.data.norm(2).item())
            optim_dict['domain_discriminator'].step()
            domain_acc_d /= len(low_features) * low_features[0].shape[0]

            # add loss to backbone
            domain_loss_g = 0
            inverse_label = 1 - label
            for l in low_features:
                d_score = model_dict['domain_discriminator'](l).squeeze()  # [4*N]
                domain_loss_g += criterion_dict['bce'](d_score, inverse_label)
            domain_loss_g *= config.LOSS.DOMAIN_LOSS_WEIGHT
            loss += domain_loss_g

        if config.LOSS.USE_HEATMAP_MI_LOSS:
            hmi_loss_d = 0
            hmi_loss_g = 0
            # iterative training
            if epoch % 2 == 0:
                low_features_detach = [l.detach() for l in low_features]
                output_detach = [o.detach() for o in output]
                for l, o, w, m in zip(low_features_detach, output_detach, weight_cuda, meta):
                    hmi_loss_d += criterion_dict['heatmap_mi'](l, o, w, m, joint_idx=config.HEATMAP_DISCRIMINATOR.JOINT_IDX)
                optim_dict['heatmap_discriminator'].zero_grad()
                hmi_loss_d.backward()
                optim_dict['heatmap_discriminator'].step()
            else:
                # if epoch == 0 and i < 1000:
                #     step_factor = 0.01
                # else:
                #     step_factor = 0.1
                for l, o, w, m in zip(low_features, output, weight_cuda, meta):
                    hmi_loss_g += criterion_dict['heatmap_mi'](l, o, w, m, joint_idx=config.HEATMAP_DISCRIMINATOR.JOINT_IDX) * config.LOSS.HEATMAP_MI_LOSS_WEIGHT
                loss += hmi_loss_g

        # consistent loss, fund loss and view mi loss
        if config.LOSS.USE_FUNDAMENTAL_LOSS or config.LOSS.USE_VIEW_MI_LOSS or (is_aggre and config.LOSS.USE_CONSISTENT_LOSS) \
            or config.LOSS.USE_JOINTS_MI_LOSS:
            s_raw_h36m, _ = select_out_h36m(raw_features, raw_features, meta)
            if len(s_raw_h36m[0]) != 0:
                s_weight_cuda, s_output = select_out_h36m(weight_cuda, output, meta)
                s_meta = select_out_h36m_meta(meta, ['center', 'scale', 'subject'])
                assert len(s_weight_cuda[0]) == len(s_output[0])
                assert len(s_weight_cuda[0]) == len(s_raw_h36m[0])

                # consistent loss on multivew h36m
                if is_aggre and config.LOSS.USE_CONSISTENT_LOSS:
                    s_agg_h36m, _ = select_out_h36m(aggre_features, aggre_features, meta)
                    cat_raw_h36m, cat_agg_h36m = torch.cat(s_raw_h36m, dim=0), torch.cat(s_agg_h36m, dim=0)
                    cal_loss = criterion_dict['mse'](cat_raw_h36m, cat_agg_h36m)
                    loss += cal_loss * config.LOSS.CONSISTENT_LOSS_WEIGHT
                    consistent_loss += cal_loss.item() * config.LOSS.CONSISTENT_LOSS_WEIGHT

                # transform heatmap to joints2d in image coordinates
                if config.LOSS.USE_FUNDAMENTAL_LOSS or config.LOSS.USE_VIEW_MI_LOSS or config.LOSS.USE_JOINTS_MI_LOSS:
                    joints2d_list = []
                    for o in s_output:
                        joints2d_list.append(generate_integral_preds_2d_th(o))
                    joints2d_list = transform_back_th(config, joints2d_list, s_meta)

                # cross-view constraints by fundamental matrix
                if config.LOSS.USE_FUNDAMENTAL_LOSS:
                    # input is a list of 4 views
                    fund_loss += criterion_dict['fundamental'](joints2d_list, s_weight_cuda, s_meta)
                    fund_loss *= config.LOSS.FUNDAMENTAL_LOSS_WEIGHT
                    loss += fund_loss

                # Mutual Information between views
                if config.LOSS.USE_VIEW_MI_LOSS:
                    view_loss_d = 0
                    view_loss_g = 0
                    # iterative training
                    if epoch % 2 == 0:
                        joints2d_list_detach = [joints2d.detach() for joints2d in joints2d_list]
                        view_loss_d += criterion_dict['view_mi'](joints2d_list_detach)
                        optim_dict['view_discriminator'].zero_grad()
                        view_loss_d.backward()
                        optim_dict['view_discriminator'].step()
                    else:
                        # if epoch == 0 and i < 1000:
                        #     step_factor = 0.01
                        # else:
                        #     step_factor = 0.1
                        view_loss_g += criterion_dict['view_mi'](joints2d_list) * config.LOSS.VIEW_MI_LOSS_WEIGHT
                        loss += view_loss_g

                # Mutual Information between joints
                if config.LOSS.USE_JOINTS_MI_LOSS:
                    jmi_loss_d = 0
                    jmi_loss_g = 0
                    # iterative training
                    if epoch % 2 == 0:
                        joints2d_list_detach = [joints2d.detach() for joints2d in joints2d_list]
                        for j in joints2d_list_detach:
                            jmi_loss_d += criterion_dict['joints_mi'](j)
                        optim_dict['joints_discriminator'].zero_grad()
                        jmi_loss_d.backward()
                        optim_dict['joints_discriminator'].step()
                    else:
                        # if epoch == 0 and i < 1000:
                        #     step_factor = 0.01
                        # else:
                        #     step_factor = 0.1
                        for j in joints2d_list:
                            jmi_loss_g += criterion_dict['joints_mi'](j, var2_no_grad=False) * config.LOSS.JOINTS_MI_LOSS_WEIGHT
                        loss += jmi_loss_g

        if config.LOSS.WATCH_GRAD_NORM:
            losses_dict = OrderedDict([('mse', mse_loss)])
            if config.LOSS.USE_FUNDAMENTAL_LOSS:
                losses_dict['fund'] = fund_loss
            if config.LOSS.USE_JOINTS_MI_LOSS and epoch % 2 != 0:
                losses_dict['jmi_g'] = jmi_loss_g
            if config.LOSS.USE_HEATMAP_MI_LOSS and epoch % 2 != 0:
                losses_dict['hmi_g'] = hmi_loss_g
            if config.LOSS.USE_VIEW_MI_LOSS and epoch % 2 != 0:
                losses_dict['vmi_g'] = view_loss_g
            grad_dict = check_grad_norm(losses_dict, raw_features, norm=1)

        # update generator
        optim_dict['base_model'].zero_grad()
        loss.backward()
        optim_dict['base_model'].step()

        # # training together
        # if config.LOSS.USE_VIEW_MI_LOSS:
        #     optim_dict['view_discriminator'].zero_grad()
        #     optim_dict['view_discriminator'].step()

        if rank == 0:
            optional_msg = ''
            losses.update(loss.item(), len(input) * input[0].size(0))
            mse_losses.update(mse_loss.item(), len(input) * input[0].size(0))

            if config.LOSS.USE_CONSISTENT_LOSS:
                consistent_losses.update(consistent_loss, len(input) * input[0].size(0))
                optional_msg += 'Consistent Loss {cons_loss.val:.5f} ({cons_loss.avg:.5f})\t'.format(
                    cons_loss=consistent_losses)

            if config.LOSS.USE_FUNDAMENTAL_LOSS:
                fund_losses.update(fund_loss.item() if isinstance(fund_loss, torch.Tensor) else fund_loss, 1)
                optional_msg += 'Fundamental Loss {fund_loss.val:.4f} ({fund_loss.avg:.4f})\t'.format(
                    fund_loss=fund_losses)

            if config.LOSS.USE_GLOBAL_MI_LOSS:
                global_mi_losses.update(global_mi_loss, 1)
                optional_msg += 'Global MI Loss {global_loss.val:.4f} ({global_loss.avg:.4f})\t'.format(
                    global_loss=global_mi_losses)
            if config.LOSS.USE_LOCAL_MI_LOSS:
                local_mi_losses.update(local_mi_loss, 1)
                optional_msg += 'Local MI Loss {local_loss.val:.4f} ({local_loss.avg:.4f})\t'.format(
                    local_loss=local_mi_losses)

            if config.LOSS.USE_DOMAIN_TRANSFER_LOSS:
                domain_losses_g.update(domain_loss_g.item(), 1)
                domain_losses_d.update(domain_loss_d.item(), 1)
                domain_accuracy_d.update(domain_acc_d.item(), 1)
                optional_msg += 'Domain_g Loss {domain_loss_g.val:.4f} ({domain_loss_g.avg:.4f})\t' \
                    'Domain_d Loss {domain_loss_d.val:.4f} ({domain_loss_d.avg:.4f})\t' \
                    'Domain_d Acc {domain_acc_d.val:.3f} ({domain_acc_d.avg:.3f})\t'.format(
                        domain_loss_g=domain_losses_g, domain_loss_d=domain_losses_d, domain_acc_d=domain_accuracy_d)

            if config.LOSS.USE_VIEW_MI_LOSS:
                if epoch % 2 == 0:
                    view_losses_d.update(view_loss_d.item() if isinstance(view_loss_d, torch.Tensor) else view_loss_d, 1)
                else:
                    view_losses_g.update(view_loss_g.item() if isinstance(view_loss_g, torch.Tensor) else view_loss_g, 1)
                optional_msg += 'View_g Loss {view_loss_g.val:.4f} ({view_loss_g.avg:.4f})\t' \
                    'View_d Loss {view_loss_d.val:.4f} ({view_loss_d.avg:.4f})\t'.format(
                        view_loss_g=view_losses_g, view_loss_d=view_losses_d)

            if config.LOSS.USE_JOINTS_MI_LOSS:
                if epoch % 2 == 0:
                    jmi_losses_d.update(jmi_loss_d.item() if isinstance(jmi_loss_d, torch.Tensor) else jmi_loss_d, 1)
                else:
                    jmi_losses_g.update(jmi_loss_g.item() if isinstance(jmi_loss_g, torch.Tensor) else jmi_loss_g, 1)
                optional_msg += 'Jmi_g Loss {jmi_loss_g.val:.4f} ({jmi_loss_g.avg:.4f})\t' \
                    'Jmi_d Loss {jmi_loss_d.val:.4f} ({jmi_loss_d.avg:.4f})\t'.format(
                        jmi_loss_g=jmi_losses_g, jmi_loss_d=jmi_losses_d)

            if config.LOSS.USE_HEATMAP_MI_LOSS:
                if epoch % 2 == 0:
                    hmi_losses_d.update(hmi_loss_d.item() if isinstance(hmi_loss_d, torch.Tensor) else hmi_loss_d, 1)
                else:
                    hmi_losses_g.update(hmi_loss_g.item() if isinstance(hmi_loss_g, torch.Tensor) else hmi_loss_g, 1)
                optional_msg += 'Hmi_g Loss {hmi_loss_g.val:.4f} ({hmi_loss_g.avg:.4f})\t' \
                    'Hmi_d Loss {hmi_loss_d.val:.4f} ({hmi_loss_d.avg:.4f})\t'.format(
                        hmi_loss_g=hmi_losses_g, hmi_loss_d=hmi_losses_d)

            if config.LOSS.WATCH_GRAD_NORM:
                mse_grad_norms.update(grad_dict['mse'].item(), 1)
                optional_msg += 'MSE_grad Norm {mse_grad_norm.val:.6f} ({mse_grad_norm.avg:.6f})\t'.format(
                    mse_grad_norm=mse_grad_norms)

                if config.LOSS.USE_FUNDAMENTAL_LOSS:
                    fund_grad_norms.update(grad_dict['fund'].item(), 1)
                    optional_msg += 'Fund_grad Norm {fund_grad_norm.val:.4f} ({fund_grad_norm.avg:.4f})\t'.format(
                        fund_grad_norm=fund_grad_norms)
                
                if config.LOSS.USE_JOINTS_MI_LOSS and epoch % 2 != 0:
                    jmi_grad_norms.update(grad_dict['jmi_g'].item(), 1)
                    optional_msg += 'Jmi_grad Norm {jmi_grad_norm.val:.5f} ({jmi_grad_norm.avg:.5f})\t'.format(
                        jmi_grad_norm=jmi_grad_norms)

                if config.LOSS.USE_HEATMAP_MI_LOSS and epoch % 2 != 0:
                    hmi_grad_norms.update(grad_dict['hmi_g'].item(), 1)
                    optional_msg += 'Hmi_grad Norm {hmi_grad_norm.val:.5f} ({hmi_grad_norm.avg:.5f})\t'.format(
                        hmi_grad_norm=hmi_grad_norms)

                if config.LOSS.USE_VIEW_MI_LOSS and epoch % 2 != 0:
                    vmi_grad_norms.update(grad_dict['vmi_g'].item(), 1)
                    optional_msg += 'Vmi_grad Norm {vmi_grad_norm.val:.5f} ({vmi_grad_norm.avg:.5f})\t'.format(
                        vmi_grad_norm=vmi_grad_norms)

            nviews = len(output)
            acc = [None] * nviews
            cnt = [None] * nviews
            pre = [None] * nviews
            for j in range(nviews):
                _, acc[j], cnt[j], pre[j] = accuracy(
                    output[j].detach().cpu().numpy(),
                    target_cuda[j].detach().cpu().numpy())
            acc = np.mean(acc)
            cnt = np.mean(cnt)
            avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'MSE Loss {mse_loss.val:.4f} ({mse_loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                          epoch, i, len(data), batch_time=batch_time,
                          speed=len(input) * input[0].size(0) / batch_time.val,
                          data_time=data_time, loss=losses, acc=avg_acc, memory=gpu_memory_usage,
                          mse_loss=mse_losses)
                msg += optional_msg
                msg += percent_of_datasource(meta)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_mse_loss', mse_losses.val, global_steps)
                writer.add_scalar('train_acc', avg_acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                if config.LOSS.USE_CONSISTENT_LOSS:
                    writer.add_scalar('train_consistent_loss', consistent_losses.val, global_steps)
                if config.LOSS.USE_FUNDAMENTAL_LOSS:
                    writer.add_scalar('train_fundamental_loss', fund_losses.val, global_steps)
                if config.LOSS.USE_GLOBAL_MI_LOSS:
                    writer.add_scalar('train_global_mi_loss', global_mi_losses.val, global_steps)
                if config.LOSS.USE_LOCAL_MI_LOSS:
                    writer.add_scalar('train_local_mi_loss', local_mi_losses.val, global_steps)
                if config.LOSS.USE_DOMAIN_TRANSFER_LOSS:
                    writer.add_scalar('train_domain_loss_g', domain_losses_g.val, global_steps)
                    writer.add_scalar('train_domain_loss_d', domain_losses_d.val, global_steps)
                    writer.add_scalar('train_domain_acc_d', domain_accuracy_d.val, global_steps)
                if config.LOSS.USE_VIEW_MI_LOSS:
                    writer.add_scalar('train_view_loss_d', view_losses_d.val, global_steps)
                    writer.add_scalar('train_view_loss_g', view_losses_g.val, global_steps)
                if config.LOSS.USE_JOINTS_MI_LOSS:
                    writer.add_scalar('train_jmi_loss_d', jmi_losses_d.val, global_steps)
                    writer.add_scalar('train_jmi_loss_g', jmi_losses_g.val, global_steps)
                if config.LOSS.USE_HEATMAP_MI_LOSS:
                    writer.add_scalar('train_hmi_loss_d', hmi_losses_d.val, global_steps)
                    writer.add_scalar('train_hmi_loss_g', hmi_losses_g.val, global_steps)

                for k in range(len(input)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{}_{:08}'.format(
                        os.path.join(output_dir, 'train'), view_name, i)
                    save_debug_images(config, input[k], meta[k], target_cuda[k],
                                      pre[k] * 4, output[k].detach(), prefix)


def validate(config,
             loader,
             dataset,
             model_dict,
             criterion_dict,
             output_dir,
             writer_dict,  # None
             rank):
    # only rank 0 process will enter this function
    device = torch.device('cuda', rank)
    for model in model_dict.values():
        model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    nsamples = len(dataset) * 4
    is_aggre = config.NETWORK.AGGRE
    njoints = config.NETWORK.NUM_JOINTS
    height = int(config.NETWORK.HEATMAP_SIZE[0])
    width = int(config.NETWORK.HEATMAP_SIZE[1])
    all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)
    all_heatmaps = np.zeros(
        (nsamples, njoints, height, width), dtype=np.float32)

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, weight, meta) in enumerate(loader):

            input = [view.to(device, non_blocking=False) for view in input]
            raw_features, aggre_features, _, _ = model_dict['base_model'](input)

            if is_aggre and config.TEST.FUSE_OUTPUT:
                output = fuse_routing(raw_features, aggre_features, is_aggre, meta)
            else:
                output = raw_features

            if config.TEST.FLIP_TEST:
                # only support MPII flip
                # input : a list of [N, 3, H, W]
                input_flipped = [torch.flip(view, dims=[3]).to(device, non_blocking=False) for view in input]
                raw_features_flipped, aggre_features_flipped, _, _ = model_dict['base_model'](input_flipped)

                if is_aggre and config.TEST.FUSE_OUTPUT:
                    output_flipped = fuse_routing(raw_features_flipped, aggre_features_flipped, is_aggre, meta)
                else:
                    output_flipped = raw_features_flipped
                output_flipped = flip_back_th(output_flipped, dataset.flip_pairs)

                if config.TEST.SHIFT_HEATMAP:
                    # in-place shift
                    for view in output_flipped:
                        view[:, :, :, 1:] = view.clone()[:, :, :, 0:-1]
                output = [(view + view_flipped)*0.5 for view, view_flipped in zip(output, output_flipped)]

            loss = 0
            target_cuda = []
            weight_cuda = []
            # loss on single view, with ground truth heat maps
            for t, w, r in zip(target, weight, raw_features):
                t = t.to(device, non_blocking=False)
                w = w.to(device, non_blocking=False)
                target_cuda.append(t)
                weight_cuda.append(w)
                loss += criterion_dict['mse_weights'](r, t, w)

            # loss on multivew h36m, consistent loss
            if is_aggre:
                if config.LOSS.USE_CONSISTENT_LOSS:
                    raw_h36m, agg_h36m = select_out_h36m(raw_features, aggre_features, meta)
                    assert len(raw_h36m[0]) == len(agg_h36m[0])
                    if len(raw_h36m[0]) != 0:
                        raw_h36m, agg_h36m = torch.cat(raw_h36m, dim=0), torch.cat(agg_h36m, dim=0)
                        loss += criterion_dict['mse'](raw_h36m, agg_h36m)

                if config.DATASET.PSEUDO_LABEL_PATH:
                    # mse loss on output, with pseudo heat maps
                    for t, w, o in zip(target_cuda, weight_cuda, output):
                        cal_loss = criterion_dict['mse_weights'](o, t, w)
                        loss += cal_loss * config.LOSS.MSE_LOSS_WEIGHT

            nimgs = len(input) * input[0].size(0)
            losses.update(loss.item(), nimgs)

            nviews = len(output)
            acc = [None] * nviews
            cnt = [None] * nviews
            pre = [None] * nviews
            for j in range(nviews):
                _, acc[j], cnt[j], pre[j] = accuracy(
                    output[j].detach().cpu().numpy(),
                    target_cuda[j].detach().cpu().numpy())
            acc = np.mean(acc)
            cnt = np.mean(cnt)
            avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            preds = np.zeros((nimgs, njoints, 3), dtype=np.float32)
            heatmaps = np.zeros(
                (nimgs, njoints, height, width), dtype=np.float32)
            for k, o, m in zip(range(nviews), output, meta):
                pred, maxval = get_final_preds(config,
                                               o.clone().cpu().numpy(),
                                               m['center'].numpy(),
                                               m['scale'].numpy())
                pred = pred[:, :, 0:2]
                pred = np.concatenate((pred, maxval), axis=2)
                preds[k::nviews] = pred
                heatmaps[k::nviews] = o.clone().cpu().numpy()

            all_preds[idx:idx + nimgs] = preds
            all_heatmaps[idx:idx + nimgs] = heatmaps
            idx += nimgs

            if i % config.PRINT_FREQ == 0 and rank == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time,
                          loss=losses, acc=avg_acc)
                logger.info(msg)

                for k in range(len(input)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{}_{:08}'.format(
                        os.path.join(output_dir, 'validation'), view_name, i)
                    save_debug_images(config, input[k], meta[k], target_cuda[k],
                                      pre[k] * 4, output[k], prefix)

        perf_indicator = 1000
        if rank == 0:
            # save heatmaps and joint locations
            u2a = dataset.u2a_mapping
            u2a = {k:v  for k, v in u2a.items() if v != '*'}
            sorted_u2a = sorted(u2a.items(), key=lambda x: x[0])
            u = np.array([mapping[0] for mapping in sorted_u2a])


            file_name = os.path.join(output_dir, 'heatmaps_locations_%s_%s.h5' % (dataset.subset, dataset.dataset_type))
            file = h5py.File(file_name, 'w')
            file['heatmaps'] = all_heatmaps[:, u, :, :]
            file['locations'] = all_preds[:, u, :]
            file['joint_names_order'] = u  # names order in union(mpii) dataset
            file.close()

            name_value, perf_indicator = dataset.evaluate(all_preds[:, u, :], output_dir if config.DEBUG.SAVE_ALL_PREDS else None)
            names = name_value.keys()
            values = name_value.values()
            num_values = len(name_value)
            _, full_arch_name = get_model_name(config)
            logger.info('| Arch ' +
                        ' '.join(['| {}'.format(name) for name in names]) + ' |')
            logger.info('|---' * (num_values + 1) + '|')
            logger.info('| ' + full_arch_name + ' ' +
                        ' '.join(['| {:.3f}'.format(value) for value in values]) +
                        ' |')

    return perf_indicator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
