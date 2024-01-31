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
import numpy as np
import torch
import datetime
from accelerate import Accelerator

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 10
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
dataset = 'compiled'
checkpoint = False

gradient_accumulation_steps = 5 * 8
batch_size = 8

block_size = 1024

# model
# 124m params
n_layer=12
n_head=12
n_embd=768

# 512m params 
# n_layer = 24
# n_head = 16
# n_embd = 1280

# 774m params
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

vocab_size = 32064  # Default vocab size
detailed_logging = False  # Set to True to enable detailed logging for each mini-batch

# Parse command-line arguments
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_batch(split, data_dir, block_size, batch_size):
    """Fetch a batch of data for training or validation."""
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x, y

def estimate_loss(model, eval_iters, accelerator, get_batch, data_dir, block_size, batch_size):
    """Estimate model loss over train/val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, block_size, batch_size)
            X, Y = accelerator.prepare(X, Y)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Calculate learning rate with linear warmup and cosine decay."""
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
log_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{dataset}.log"
log_filepath = os.path.join(out_dir, log_filename)

def log(message):
    """Log a message to both the console and the log file."""
    if accelerator.is_main_process:
        print(message)
        with open(log_filepath, 'a') as f:
            f.write(f'{message}\n')

# -----------------------------------------------------------------------------
# Setup and Initialization
# -----------------------------------------------------------------------------
accelerator = Accelerator(mixed_precision='fp16')  # Enable mixed precision (FP16)
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
log(f"Tokens per iteration will be: {tokens_per_iter:,}")
data_dir = os.path.join('data', dataset)

# Initialize best validation loss to infinity
best_val_loss = float('inf')

if accelerator.is_main_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Model Initialization
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout
)

# Resume from a checkpoint if specified
epoch = 0
if init_from == 'scratch':
    log("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    checkpoint = checkpoint if checkpoint else os.path.join(out_dir, 'ckpt.pt')
    log(f"Resuming training from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location='cpu')
    for k in checkpoint['model_args']:
        model_args[k] = checkpoint['model_args'][k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    # Remove unwanted layers
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(checkpoint['model'])

    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

else:
    raise ValueError("Invalid init_from value. Choose 'scratch' or 'resume'.")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model = accelerator.prepare(model)
optimizer = GPT.configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), 'cuda')
# Load optimizer state if resuming from checkpoint
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
optimizer = accelerator.prepare(optimizer)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

while epoch < max_iters:
    X, Y = get_batch('train', data_dir, block_size, batch_size)
    X, Y = accelerator.prepare(X, Y)
    t0 = time.time()
    lr = get_lr(epoch, warmup_iters, lr_decay_iters, learning_rate, min_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if epoch % eval_interval == 0 and accelerator.is_main_process:
        losses = estimate_loss(model, eval_iters, accelerator, get_batch, data_dir, block_size, batch_size)
        log(f"Epoch {epoch}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }
            checkpoint_path = os.path.join(out_dir, f'ckpt_epoch_{epoch}_loss_{best_val_loss}.pt')
            accelerator.save(checkpoint, checkpoint_path)
            log(f"Checkpoint saved to {checkpoint_path}")

    if eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        if grad_clip != 0.0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if detailed_logging and accelerator.is_main_process:
            log(f"Epoch {epoch}, Step {micro_step}: Loss {loss.item()}")

        # Prepare the next batch
        X, Y = get_batch('train', data_dir, block_size, batch_size)
        X, Y = accelerator.prepare(X, Y)

    dt = time.time() - t0
    if epoch % log_interval == 0 and accelerator.is_main_process:
        lossf = loss.item() * gradient_accumulation_steps
        log(f"Epoch {epoch}: Loss {lossf:.4f}, Time {dt*1000:.2f}ms")
    epoch += 1
