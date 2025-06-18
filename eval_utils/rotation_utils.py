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
    """
    Applies a learned rotation matrix (R1) to the output projection layer (lm_head)
    of the model. This aligns the final output layer with earlier rotated representations
    (e.g., from attention or MLP layers), maintaining consistency in the rotated space.

    Args:
    - model (torch.nn.Module): The model containing the lm_head to be rotated.
    - R1 (torch.Tensor): The learned rotation matrix used to transform the output weights.
    """
    # Get the language modeling head (linear layer projecting to vocabulary size)
    W = model.lm_head

    # Preserve original data type to cast back after rotation
    dtype = W.weight.data.dtype

    # Move weights to GPU and upcast to float64 for more accurate matrix multiplication
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)

    # Apply the learned rotation matrix to the weights: W = W @ R1
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    """
    Applies a structured Hadamard-based rotation (optionally using a learned matrix R2)
    to the value projection (v_proj) and output projection (o_proj) matrices of the 
    self-attention layer in a transformer block.

    Args:
        layer: A single transformer block containing the self-attention module.
        head_num (int): Number of attention heads.
        head_dim (int): Dimension of each attention head (typically hidden_size / head_num).
        R2 (torch.Tensor, optional): Optional learned rotation matrix (Hadamard-like).
                                     If provided, it is used instead of a fixed Hadamard matrix.

    Notes:
        - `v_proj` projects input hidden states to values in attention.
        - `o_proj` projects concatenated outputs of all heads to the next layer.
        - `apply_exact_had_to_linear` performs Hadamard or R2-based structured transformation.
        - `output=True` for `v_proj` implies transposition during Hadamard application.
        - `output=False` for `o_proj` applies the inverse-like rotation.
    """

    # Get the value projection (V) and output projection (O) layers from self-attention
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    # Apply Hadamard (or R2) transformation on v_proj weights across the head dimension.
    # This is done with a transpose (output=True) because V weights will be used in dot-products.
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)

    # Apply Hadamard (or R2) transformation on o_proj weights (inverse direction).
    # No transpose here since the projection happens after concatenation.
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


@torch.inference_mode()  # Disables gradient tracking to reduce memory usage and improve performance during inference or model modification. 
                         # Ensures that all tensor operations inside this function do not build computation graphs (no autograd), 
                         # which is appropriate here since we're only modifying model weights, not training.
                         # It’s more memory-efficient and faster than wrapping code in with torch.no_grad().
@torch.inference_mode()
def rotate_model(model, args):
    """
    Applies structured rotation to various weights of the transformer model.

    This function handles both:
    - The generation and application of orthogonal (R1, R2) rotation matrices.
    - Loading pre-optimized rotation matrices (R1 and per-layer R2) if available.

    The rotation is applied to:
        - Token embeddings
        - Self-attention weights (Q, K, V, O)
        - MLP weights (input, output)
        - Final LM head
        - Value/output projections (with Hadamard or learned R2)

    Args:
        model: The model to rotate.
        args: An argument object with configuration attributes such as:
              - rotate_mode: 'hadamard' or 'random'
              - optimized_rotation_path: path to precomputed R1, R2 matrices (optional)
    
    Notes:
        - R1 is a global rotation (same for all layers), applied on model_dim.
        - R2 is a per-layer attention head dimension rotation (head_dim).
        - Uses double precision during rotation for numerical accuracy.
        - Operates in inference mode to prevent gradients and save memory.
    """
    
    # Generate or load R1: an orthogonal/global rotation matrix for hidden_size
    R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    if args.optimized_rotation_path is not None:
        # Load pre-optimized R1 matrix from checkpoint if available
        R_cpk = args.optimized_rotation_path
        R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
    
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads  # Dimension per attention head

    # Apply R1 rotation to embeddings and LM head
    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    utils.cleanup_memory()  # Free up CUDA memory after heavy operations

    # Go through each transformer layer and apply all necessary rotations
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        
        # If optimized R2 is available, load it per layer from checkpoint
        if args.optimized_rotation_path is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
        else:
            # Else, generate a fresh orthogonal R2 for this layer (head_dim x head_dim)
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)

        # Apply R1-based rotations to Q, K, V input matrices and O output matrix
        rotate_attention_inputs(layer, R1)
        rotate_attention_output(layer, R1)

        # Apply R1-based rotations to MLP up_proj/gate_proj and down_proj
        rotate_mlp_input(layer, R1)
        rotate_mlp_output(layer, R1)

        # Apply R2 (Hadamard or learned) to V and O projections with structure
        rotate_ov_proj(layer, num_heads, head_dim, R2=R2)


class QKRotationWrapper(torch.nn.Module):
    """
    A wrapper module that applies Hadamard-based rotation to Q and K matrices
    and then performs quantization on the K matrix for efficient KV caching.

    - Q is rotated using a Hadamard matrix (no quantization).
    - K is rotated and then quantized using either token-wise or head-wise granularity.
    
    Args:
        func: A callable (usually attention sub-function) that returns (Q, K).
        config: Model configuration with attention head and hidden size.
        k_bits: Number of bits for K quantization (default: 16).
        k_groupsize: Determines the quantization granularity:
                     -1 for token-wise, `head_dim` for head-wise.
        k_sym: Whether to use symmetric quantization.
        k_clip_ratio: Ratio to clip activations during quantization.
    
    Restrictions:
        - head_dim must be a power of 2 (needed for Hadamard transform).
        - Only supports token-wise or head-wise quantization for K.
    """
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config

        # Extract number of attention heads and model dimensionality
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads

        # Hadamard only works with power-of-2 dimensions
        assert is_pow2(head_dim), "Only power of 2 head_dim is supported for K-cache Quantization!"

        self.func = func  # original function to compute q, k

        # Initialize quantizer module
        self.k_quantizer = quant_utils.ActQuantizer()

        # Default number of bits for K quantization
        self.k_bits = 16

        if kwargs is not None:
            # Group size for quantization (-1 for token-wise, head_dim for head-wise)
            assert kwargs["k_groupsize"] in [-1, head_dim], "Only token-wise or head-wise quantization is supported"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]

            # Configure quantizer (groupsize=-1 forces token-wise; head-wise handled manually)
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # always treat as token-wise in module; we handle shape ourselves
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        # Get Q and K from the wrapped function
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype

        # Apply Hadamard transform (feature rotation) and normalize
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)

        # original shape of k
        bsz, num_heads, seq_len, head_dim = k.shape

        if self.k_groupsize == -1:
            # === Token-wise quantization ===
            # Reshape to [batch * seq, num_heads * head_dim] for token-level quant
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)

            # Find quantization parameters (scale, zero-point, clipping)
            self.k_quantizer.find_params(token_wise_k)

            # Quantize and reshape back to original layout
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:
            # === Head-wise quantization ===
            # Flatten head dimensions to quantize independently per head
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)

            # Quantize and reshape back
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        # Clean up to release memory (remove stored quant params)
        self.k_quantizer.free()

        # Return rotated Q and quantized+rotated K
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

    # Use monkeypatch utility to add the wrapper after the function call in the forward method.
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",              #patch the forward method
        function_name,          #target function inside forward
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )

    # Store the wrapper in the module so it can be accessed or inspected later if needed
    setattr(module, attr_name, wrapper)
