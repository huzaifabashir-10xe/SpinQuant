# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    """
    Applies a rotation (R1) to the token embedding matrix of the model.

    This is part of SpinQuant's preprocessing, where a learned or predefined
    rotation matrix R1 is applied to the input embeddings to improve quantization performance.

    Args:
        model: The LLaMA model whose embeddings will be rotated.
        R1 (torch.Tensor): A rotation matrix of shape [d, d], where d is the embedding dimension.
    """
    # Iterate over embedding matrices — here, only 'embed_tokens' is considered.
    for W in [model.model.embed_tokens]:
        # Save the original data type (e.g., float16 or bfloat16)
        dtype = W.weight.data.dtype

        # Move the embedding weights to CUDA and cast to float64 for better numerical precision
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)

        # Apply rotation: new_embeddings = W_original @ R1
        rotated = torch.matmul(W_, R1)

        # Move rotated weights back to CPU and original dtype
        W.weight.data = rotated.to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    """
    Applies a rotation matrix (R1) to the input projection matrices (Q, K, V)
    of the self-attention mechanism in a transformer layer.

    This is part of SpinQuant's preprocessing phase where rotation (R1)
    is applied to attention input weights to structure the input feature space
    in a way that is more quantization-friendly.

    Args:
        layer: A single transformer block/layer (e.g., LLaMA block) containing a `self_attn` module.
        R1 (torch.Tensor): Rotation matrix of shape [d, d], where d is the model's hidden size.
    """
    # Apply R1 to each of the attention input projections: W_Q, W_K, W_V
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        # Save the original data type (e.g., float16 or bfloat16)
        dtype = W.weight.dtype

        # Move weight to CUDA and cast to float64 for numerical precision
        W_ = W.weight.to(device="cuda", dtype=torch.float64)

        # Apply rotation: W_rotated = W @ R1
        rotated = torch.matmul(W_, R1)

        # Move back to CPU and restore original dtype
        W.weight.data = rotated.to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    """
    Applies the transpose of the rotation matrix R1 to the output projection
    matrix of the self-attention layer (W_O) and its bias, effectively reversing
    the earlier rotation applied to Q, K, V, and embeddings.

    This ensures that the overall function of the model remains the same after
    the initial R1 rotation, completing the similarity transformation:
    Output = R1ᵀ · (Original_Transform · R1 · Input)

    Args:
        layer: A transformer layer (e.g., from LLaMA) that contains a self-attention module.
        R1 (torch.Tensor): The same rotation matrix applied earlier to Q, K, V and embeddings.
    """

    # Extract the output projection (W_O) of the attention block
    W = layer.self_attn.o_proj

    # Preserve the original data type (e.g., float16 or bfloat16)
    dtype = W.weight.data.dtype

    # Move weight to CUDA and upcast to float64 for numerical precision
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)

    # Apply the inverse rotation (R1ᵀ @ W_O)
    rotated_weight = torch.matmul(R1.T, W_)

    # Move rotated weight back to CPU and cast back to original dtype
    W.weight.data = rotated_weight.to(device="cpu", dtype=dtype)

    # If a bias term exists for W_O, rotate it as well
    if W.bias is not None:
        # Move bias to CUDA and upcast
        b = W.bias.data.to(device="cuda", dtype=torch.float64)

        # Rotate bias using R1ᵀ (same as for weights)
        rotated_bias = torch.matmul(R1.T, b)

        # Cast and move back
        W.bias.data = rotated_bias.to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    """
    Applies rotation matrix R1 to the input projection weights of the MLP (feed-forward) block
    in a transformer layer. This includes `up_proj` and `gate_proj`.

    This rotation transforms the input activations of the MLP block, aligning them with
    the rotated embedding and attention space.

    Args:
        layer: A transformer layer containing an MLP block with `up_proj` and `gate_proj`.
        R1 (torch.Tensor): The rotation matrix applied to embeddings and attention inputs.
    """

    # Get the input projection layers of the MLP
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]

    for W in mlp_inputs:
        # Store the original data type (bfloat16, float16, etc.)
        dtype = W.weight.dtype

        # Move the weight to CUDA and upcast to float64 for stable matrix multiplication
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)

        # Apply the rotation matrix: W_rotated = W_original @ R1
        rotated_weight = torch.matmul(W_, R1)

        # Move the result back to CPU and convert to original dtype
        W.weight.data = rotated_weight.to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1):
    """
    Applies the transpose of rotation matrix R1 to the output projection (down_proj) 
    weights and bias of the MLP (feed-forward) block in a transformer layer.

    This effectively reverses the earlier input-space rotation (by applying R1.T),
    ensuring the output is mapped back to the original representation space.

    Additionally, applies an exact inverse Hadamard transformation to the weights 
    for improved quantization properties.

    Args:
        layer: A transformer layer containing an MLP block with `down_proj`.
        R1 (torch.Tensor): The rotation matrix used for embedding and attention space alignment.
    """

    # Access the down projection layer of the MLP block
    W = layer.mlp.down_proj

    # Store original data type (e.g., float16, bfloat16)
    dtype = W.weight.data.dtype

    # Move the weights to CUDA and upcast to float64 for numerical stability
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)

    # Apply inverse rotation: W_rotated = R1.T @ W_original
    # This maps the rotated activations back to the original space
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)

    # Apply an exact Hadamard transformation on the output projection weights
    # This prepares weights for Hadamard-aware quantization
    apply_exact_had_to_linear(W, had_dim=-1, output=False)

    # Rotate the bias term if it exists, same as the weights
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


@torch.inference_mode()
def rotate_model(model, args):
    R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if args.optimized_rotation_path is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
        else:
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
        rotate_attention_inputs(layers[idx], R1)
        rotate_attention_output(layers[idx], R1)
        rotate_mlp_input(layers[idx], R1)
        rotate_mlp_output(layers[idx], R1)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
