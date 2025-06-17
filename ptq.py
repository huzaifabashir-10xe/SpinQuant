# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")


def train() -> None:
    """
    Perform post-training quantization (PTQ) evaluation on a pretrained LLaMA model.
    This function sets up distributed environment, loads the model and tokenizer, 
    applies PTQ transformations (like rotation and quantization), and evaluates the 
    model on the WikiText-2 dataset using perplexity as the metric.
    """

    # Initialize distributed training group with NCCL backend
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    
    # Parse command-line/model/training arguments it includes arguments for model, training parameters and quantization including roation
    model_args, training_args, ptq_args = process_args_ptq()

    #local_rank tells each process which GPU it should use on the current machine.    
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))

    # Synchronize all processes before proceeding
    torch.distributed.barrier()

    # Load model configuration
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    
    # Set torch dtype to bfloat16 or float16
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
    # Load pretrained LLaMA model with custom config and dtype
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )

    # If process_word_embeddings is enabled above, manually copy weights from embed_tokens to lm_head
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # Move model to GPU    
    model.cuda()

    # Apply Post-Training Quantization + Rotation logic (SpinQuant)
    model = ptq_model(ptq_args, model, model_args)
    
    # Set model sequence length
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    
    # Load tokenizer for input tokenization
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

    # Disable KV caching in forward pass to ensure fresh computation (needed for rotation)
    model.config.use_cache = False

    # Load evaluation data: WikiText-2 in inference mode
    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    # Run evaluation using perplexity on the testloader
    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))

    # Synchronize all processes at end of run    
    dist.barrier()


if __name__ == "__main__":
    train()
