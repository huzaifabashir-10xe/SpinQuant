# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is originally from: https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/stiefel_optimizer.py

import random

import torch
from torch.optim.optimizer import Optimizer


def unit(v, dim: int = 1, eps: float = 1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def norm(v, dim: int = 1):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out


def Cayley_loop(X, W, tan_vec, t):  #
    [n, p] = X.size()
    Y = X + t * tan_vec
    for i in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))

    return Y.t()


import torch

# Small constant to avoid division by zero in normalization
episilon = 1e-8


def unit(v, dim: int = 1, eps: float = 1e-8):
    """
    Normalizes a 2D tensor along the specified dimension.

    Args:
        v (Tensor): Input tensor of shape (n, m).
        dim (int): Dimension to normalize along. Default is 1 (columns).
        eps (float): Small constant to avoid division by zero.

    Returns:
        Tuple[Tensor, Tensor]: Tuple of (normalized tensor, norms before normalization).
    """
    vnorm = norm(v, dim)             # Compute L2 norm along given dimension
    return v / vnorm.add(eps), vnorm # Normalize and return both normalized vector and norms


def norm(v, dim: int = 1):
    """
    Computes the L2 norm of a 2D tensor along a given dimension.

    Args:
        v (Tensor): Input 2D tensor.
        dim (int): Dimension to reduce along.

    Returns:
        Tensor: Norms with dimensions preserved (keepdim=True).
    """
    assert len(v.size()) == 2, "Input tensor must be 2-dimensional"
    return v.norm(p=2, dim=dim, keepdim=True)


def matrix_norm_one(W):
    """
    Computes the matrix 1-norm (maximum absolute column sum).

    Args:
        W (Tensor): Input 2D matrix.

    Returns:
        Tensor: Scalar tensor representing the matrix 1-norm.
    """
    out = torch.abs(W)           # Take element-wise absolute value
    out = torch.sum(out, dim=0)  # Sum across rows (i.e., column sums)
    out = torch.max(out)         # Take the maximum column sum
    return out


def Cayley_loop(X, W, tan_vec, t):
    """
    Performs iterative Cayley retraction for Stiefel manifold optimization.

    Cayley transform preserves orthogonality. The loop refines an update step
    that moves `X` along a skew-symmetric direction `W`.

    Args:
        X (Tensor): Current orthonormal matrix of shape (n, p).
        W (Tensor): Skew-symmetric update matrix of shape (n, n).
        tan_vec (Tensor): Tangent vector (typically momentum) of shape (n, p).
        t (float): Step size (learning rate scaled).

    Returns:
        Tensor: Transposed updated matrix of shape (p, n).
    """
    [n, p] = X.size()
    Y = X + t * tan_vec  # Initial update guess

    # Refine Y iteratively using midpoint formulation
    for i in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))

    return Y.t()  # Return transposed result to match caller expectations


def qr_retraction(tan_vec):
    """
    Projects the input matrix onto the Stiefel manifold using QR decomposition.

    Args:
        tan_vec (Tensor): Tangent vector matrix of shape (p, n), where p ≤ n.

    Returns:
        Tensor: Orthonormal matrix of shape (p, n) such that rows form an orthonormal set.
    """
    [p, n] = tan_vec.size()
    tan_vec.t_()  # Transpose to shape (n, p) for QR decomposition

    q, r = torch.linalg.qr(tan_vec)  # QR decomposition

    d = torch.diag(r, 0)             # Extract diagonal of R
    ph = d.sign()                    # Compute sign of diagonal entries

    q *= ph.expand_as(q)            # Adjust sign to ensure consistency
    q.t_()                          # Transpose back to original shape (p, n)

    return q


episilon = 1e-8


class SGDG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'.

        If stiefel is True, the variables will be updated by SGD-G proposed
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(
        self,
        params,
        lr,
        momentum: int = 0,
        dampening: int = 0,
        weight_decay: int = 0,
        nesterov: bool = False,
        stiefel: bool = False,
        omega: int = 0,
        grad_clip=None,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            omega=0,
            grad_clip=grad_clip,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Reshape weight matrix into 2D for processing
                unity, _ = unit(p.data.view(p.size()[0], -1))

                # Check whether Stiefel update is appropriate
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    weight_decay = group["weight_decay"]
                    dampening = group["dampening"]
                    nesterov = group["nesterov"]

                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = qr_retraction(unity)

                    g = p.grad.data.view(p.size()[0], -1)

                    lr = group["lr"]

                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(g.t().size())
                        if p.is_cuda:
                            param_state["momentum_buffer"] = param_state[
                                "momentum_buffer"
                            ].cuda()

                    V = param_state["momentum_buffer"]
                    V = momentum * V - g.t()

                    # Compute skew-symmetric update matrix
                    MX = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()

                    # Compute Cayley step size
                    t = 0.5 * 2 / (matrix_norm_one(W) + episilon)
                    alpha = min(t, lr)

                    # Cayley retraction update
                    p_new = Cayley_loop(unity.t(), W, V, alpha)
                    V_new = torch.mm(W, unity.t())  # n-by-p
                    #                     check_identity(p_new.t())
                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)

                else:

                    # Standard SGD update
                    d_p = p.grad.data
                    #  defined.
                    try:
                        if weight_decay != 0:
                            #  defined.
                            d_p.add_(weight_decay, p.data)
                    except:
                        pass
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            #  always defined.
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        #  defined.
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                    
                    # Standard SGD weight update
                    p.data.add_(-group["lr"], d_p)

        return loss
