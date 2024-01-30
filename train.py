"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from datetime import datetime
import numpy as np
import torch
from accelerate import Accelerator
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default config values designed to train a GPT-2 (124M) model on OpenWebText
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 10
log_interval = 1
eval_iters = 200
eval_only = False # Set to True to exit after the first evaluation
always_save_checkpoint = True
init_from = 'scratch'
dataset = 'compiled'
gradient_accumulation_steps = 5*8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
#config for 124m paras
n_layer=12
n_head=12
n_embd=768

# config for 512m paras 
# n_layer = 24
# n_head = 16
# n_embd = 1280

# #config 774m paras
# n_layer = 36
# n_head = 20
# n_embd = 1280

dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = True # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 100 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 100 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Prepare logging
# -----------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = f'log_{dataset}_{timestamp}.log'
log_filepath = os.path.join(out_dir, log_filename)

def log_message(message):
    print(message)
    with open(log_filepath, 'a') as f:
        f.write(message + '\n')

# -----------------------------------------------------------------------------
# Setup Accelerator
# -----------------------------------------------------------------------------
accelerator = Accelerator()

# -----------------------------------------------------------------------------
# Data Loading Function
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory(), y.pin_memory()
    return x, y

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9
vocab_size = 32064

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)

if init_from == 'scratch':
    log_message("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    log_message(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model = accelerator.prepare(model)

# -----------------------------------------------------------------------------
# Optimizer Initialization
# -----------------------------------------------------------------------------
optimizer = GPT.configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), 'cuda')
optimizer = accelerator.prepare(optimizer)

# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = accelerator.prepare(X, Y, non_blocking=True)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
X, Y = accelerator.prepare(X, Y, non_blocking=True)
t0 = time.time()

while iter_num < max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and accelerator.is_main_process:
        losses = estimate_loss()
        log_message(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            checkpoint_path = os.path.join(out_dir, f'ckpt_{timestamp}.pt')
            accelerator.save(checkpoint, checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")

    if eval_only and iter_num == 0:
        break

    for micro_step in range(gradient_accumulation_steps):
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        X, Y = accelerator.prepare(X, Y)
        accelerator.backward(loss)

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and accelerator.is_main_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        log_message(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

log_message("Training complete.")
