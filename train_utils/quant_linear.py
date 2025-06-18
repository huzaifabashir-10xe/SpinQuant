# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._tensor import Tensor


class QuantizeLinear(nn.Linear):
    """
    Custom Linear layer that supports:
    - Optional application of learned rotation matrices (R1 and R2) to weights.
    - Optional weight quantization (e.g., via RTN or other quantizer with find_params + quantize interface).
    
    This is primarily used for **post-training quantization (PTQ)** where we transform and compress weights
    without retraining.
    
    If quantizer is attached (e.g., via `self.quantizer`), the rotated weights are quantized before matmul.

    Args:
        input (Tensor): Input tensor to the linear layer.
        R1 (Tensor, optional): Rotation matrix
        R2 (Tensor, optional): Rotation matrix, usually per-head block.
        transpose (bool): If True, applies transpose logic suitable for attention output projections.
    
    Returns:
        Tensor: Result of linear transformation with (optional) rotations and quantization.
    """

    def forward(
        self,
        input: Tensor,
        R1=None,
        R2=None,
        transpose=False,
    ) -> Tensor:
        # Start from original weight
        weight = self.weight

        # Apply R1 rotation (on right)
        if R1 is not None:
            dtype = weight.dtype
            if not transpose:
                # W_rotated = W @ R1
                weight = (weight.to(torch.float64) @ R1.to(torch.float64)).to(dtype)
            else:
                # W_rotated = R1.T @ W
                weight = (R1.T.to(torch.float64) @ weight.to(torch.float64)).to(dtype)

            # Apply R2 rotation (on left) in a block-wise fashion
            if R2 is not None:
                had_dim = R2.shape[0]
                dtype = weight.dtype

                if transpose:
                    # Head-wise left rotation for transposed weights
                    W_ = weight
                    init_shape = W_.shape
                    temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(init_shape)
                else:
                    # Right rotation then head-wise left rotation for normal matmul
                    W_ = weight.t()
                    transposed_shape = W_.shape
                    temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(transposed_shape).t()
                
                weight = weight.to(dtype)

        # Apply quantization if quantizer is attached (e.g., RTN)
        if hasattr(self, "quantizer"):
            dtype = weight.dtype
            self.quantizer.find_params(weight.data)
            weight = self.quantizer.quantize(weight).to(dtype)

        # Standard linear operation with rotated + quantized weights
        return nn.functional.linear(input, weight, self.bias)
