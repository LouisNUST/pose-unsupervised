# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import autograd
from collections import OrderedDict

def check_grad_norm(losses_dict, x, norm=1):
    """
    Compare the grad norm of different losses w.r.t x
    x is a list
    """

    results = OrderedDict()
    if not isinstance(x, (list, tuple)):
        x = [x]

    for name, loss in losses_dict.items():
        grad_list = autograd.grad(outputs=loss, inputs=x,
                grad_outputs=torch.ones_like(loss),
                create_graph=False, retain_graph=True,
                only_inputs=True, allow_unused=True)
        view_norm = 0
        for idx, grad in enumerate(grad_list):
            if grad is None:
                print('grad of {} (view {}) is None'.format(name, idx))
            else:
                grad = grad.view(grad.shape[0], -1)
                row_norm = grad.norm(p=norm, dim=1)
                view_norm += row_norm.sum() / row_norm.ne(0).to(dtype=row_norm.dtype).sum()
        results[name] = view_norm.mean()
    return results
