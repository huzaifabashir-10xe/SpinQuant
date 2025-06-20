# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaTokenizerFast, Trainer, default_data_collator
import transformers
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import random_hadamard_matrix
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict

log: Logger = get_logger("spinquant")


class RotateModule(nn.Module):
    """
    A PyTorch module that applies a learned rotation matrix to an input tensor.

    Args:
        R_init (torch.Tensor): The initial rotation matrix. This will be converted 
                               to a trainable parameter of type float32 and moved to CUDA.

    Attributes:
        weight (nn.Parameter): A learnable rotation matrix.

    Forward Args:
        x (torch.Tensor): Input tensor to be rotated.
        transpose (bool, optional): If True, applies x @ weight (i.e., applies rotation to the right).
                                    If False (default), applies weight @ x (i.e., applies rotation to the left).

    Returns:
        torch.Tensor: The rotated tensor after matrix multiplication.
    """
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


def train() -> None:

    # Initialize the distributed training process using NCCL backend    
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    # Parse model, training, and PTQ (Post-Training Quantization) arguments
    model_args, training_args, ptq_args = process_args_ptq()
    
    #local_rank tells each process which GPU it should use on the current machine.
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()
    #load model config
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )

    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    # Set the appropriate data type
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    # Load the quantized LLaMA model with the specified dtype
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    # If process word embeddings is enabled above, copy embedding weights to lm_head manually
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    
    # Prepare model for quantization-aware rotation
    model = prepare_model(ptq_args, model)
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Insert rotation R1 (applied before feed-forward in each layer)
    R1 = random_hadamard_matrix(model.config.hidden_size, "cuda")
    model.R1 = RotateModule(R1)
    
    # Insert per-head rotation R2 in each self-attention module
    for i in range(model.config.num_hidden_layers):
        # Each head dim = 128 for Llama model
        R2 = random_hadamard_matrix(
            model.config.hidden_size // model.config.num_attention_heads, "cuda"
        )
        model.model.layers[i].self_attn.R2 = RotateModule(R2)
    # Log model loading completion on rank 0
    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    
    # Load the tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    # Disable caching for forward pass
    model.config.use_cache = False
    # Load calibration dataset for learning rotations
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    # Collect the trainable rotation parameters (R1 and all R2s)
    trainable_parameters = [model.R1.weight] + [
        model.model.layers[i].self_attn.R2.weight
        for i in range(model.config.num_hidden_layers)
    ]

    # Set sequence length
    model.seqlen = training_args.model_max_length
    # Use custom optimizer (Cayley SGD on the Stiefel manifold for orthogonal matrices)
    optimizer = SGDG(trainable_parameters, lr=training_args.learning_rate, stiefel=True)
    MyTrainer = Trainer
    # Use FSDP (Fully Shared Data parallel) for 70B rotation training i.e train by splitting across GPUs
    if training_args.fsdp != "" and training_args.fsdp != []:
        MyTrainer = FSDPTrainer

    # Initialize the trainer
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )

    # Synchronize processes before training i.e make sure all processes wait until every process reaches that point in case of multi-gpu
    torch.distributed.barrier()

    # Begin training
    trainer.train()

    # Save rotation matrices after training (only on rank 0)
    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(trainer.model)
    else:
        cpu_state = trainer.model.state_dict()

    # Extract R1 and R2 weights from state dict
    R_dict = {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if "R1.weight" in key or "self_attn.R2" in key
    }

    # Save rotation matrices to disk
    if local_rank == 0:
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        path = os.path.join(model_args.output_rotation_path, "R.bin")
        torch.save(
            R_dict,
            path,
        )
    
    # Final synchronization
    dist.barrier()


if __name__ == "__main__":
    train()
