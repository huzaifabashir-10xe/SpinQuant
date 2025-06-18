# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import tqdm

from train_utils.quant_linear import QuantizeLinear
from utils import quant_utils, utils


@torch.no_grad()  # Disable gradient calculation to save memory and improve speed
def rtn_fwrd(model, dev, args):
    """
    RTN (Round-To-Nearest) post-training weight quantization.
    Attaches a weight quantizer to all Linear and QuantizeLinear layers.
    This follows the RTN technique from the GPTQ repo.
    """
    layers = model.model.layers  # Access transformer layers
    torch.cuda.empty_cache()  # Clear any cached GPU memory

    quantizers = {}  # Dictionary to store all quantizer objects by layer name

    # Loop through each transformer layer
    for i in tqdm.tqdm(range(len(layers)), desc="Inserting weight quantizer"):
        layer = layers[i].to(dev)  # Move layer to device (e.g., GPU)

        # Find all Linear and QuantizeLinear submodules in this layer
        subset = quant_utils.find_qlayers(
            layer, layers=[torch.nn.Linear, QuantizeLinear]
        )

        # For each linear submodule, configure and attach a quantizer
        for name in subset:
            layer_weight_bits = args.w_bits  # Default bit-width from args

            # lm_head is not quantized — use 16-bit
            if "lm_head" in name:
                layer_weight_bits = 16
                continue

            # Optionally quantize down_proj layer to 8 bits
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8

            # Create and configure a weight quantizer
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                bits=layer_weight_bits,
                perchannel=True,                      # Per-channel quantization
                sym=not (args.w_asym),                # Symmetric or asymmetric quant
                mse=args.w_clip,                      # Whether to use MSE-based clipping
                weight_groupsize=args.w_groupsize,    # Group size for weight quantization
            )

            # Attach the quantizer to the module
            subset[name].quantizer = quantizer

            # Save quantizer instance in dict with full layer path as key
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()

        # Move layer back to CPU to save GPU memory
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()  # Free any GPU memory used
        del layer  # Delete the temporary variable

    # Final memory cleanup
    utils.cleanup_memory(verbos=True)

    return quantizers  # Return all quantizers (useful for saving, logging, etc.)
