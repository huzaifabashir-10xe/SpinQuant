# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot (https://github.com/spcl/QuaRot/tree/main/quarot),
# licensed under the Apache License 2.0.

import transformers

from train_utils import apply_r3_r4, rtn_utils
from utils import fuse_norm_utils, hadamard_utils, quant_utils, utils


def prepare_model(args, model):
    """
    Prepares a transformer model for quantization with learned rotation matrices.

    This function performs the following:
    - Fuses LayerNorm layers for efficiency.
    - Applies online rotations (R3, R4) to various parts of the model.
    - Enables online Hadamard transforms in feed-forward paths (e.g., down_proj).
    - Adds activation quantization wrappers throughout the model.
    - Configures quantization parameters for attention projections (v_proj, o_proj),
      MLP (down_proj), and optionally for key cache (k).
    - Injects QKRotationWrapper for Q/K quantization post-RoPE.

    This is a **training-time** preparation step. It does not finalize static quantized weights
    or convert the model for deployment. Instead, it prepares the model to learn
    to be robust under quantization effects.

    Args:
        args: A namespace or config object containing quantization and rotation settings.
        model: A HuggingFace-style transformer model.
    
    Returns:
        The modified model ready for PTQ with rotations.
    """
    transformers.set_seed(args.seed)
    model.eval()

    # Step 1: Fuse LayerNorms (e.g., residual + norm layers)
    fuse_norm_utils.fuse_layer_norms(model)

    # Step 2: Apply rotations R3, R4 to weights and activation paths
    apply_r3_r4.rotate_model(model, args)
    utils.cleanup_memory(verbos=True)

    # Step 3: Insert activation quantization wrappers throughout model
    quant_utils.add_actquant(model)

    # Step 4: Enable online Hadamard transformations in down_proj layers
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had

    # Step 5: Apply RTN for weight quantization (if w_bits < 16)
    if args.w_bits < 16:
        quantizers = rtn_utils.rtn_fwrd(model, "cuda", args)

    # Step 6: Configure input (activation) quantization if enabled
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

        for name in qlayers:
            # Default input quant settings
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not args.a_asym
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            # Special case: v_proj gets its own settings
            if "v_proj" in name and args.v_bits < 16:
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not args.v_asym,
                    clip_ratio=args.v_clip_ratio,
                )

            # o_proj is grouped by head dimension
            if "o_proj" in name:
                layer_groupsize = head_dim

            # Skip quantizing lm_head for now
            if "lm_head" in name:
                layer_input_bits = 16

            # down_proj might be forced to int8
            if "down_proj" in name:
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            # Apply config to input quantizer
            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    # Step 7: Add QKRotationWrapper post-RoPE if k-bit quantization is enabled
    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not args.k_asym,
                "k_clip_ratio": args.k_clip_ratio,
            }
            for layer in layers:
                apply_r3_r4.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )

    return model
