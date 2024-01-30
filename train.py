"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32

To run with DDP on 8 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=8 train.py

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
import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from multiprocessing import cpu_count

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 10
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
dataset = 'compiled'
gradient_accumulation_steps = 5*8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# Model Size

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

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Initialize Accelerator
# -----------------------------------------------------------------------------
accelerator = Accelerator()

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
log_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{dataset}.log"
log_filepath = os.path.join(out_dir, log_filename)

def log(message):
    if accelerator.is_main_process:
        print(message)
        with open(log_filepath, 'a') as f:
            f.write(f'{message}\n')

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
class MemMapDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + 1 + self.block_size]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Determine the number of CPU cores
num_cpus = cpu_count()

# Set num_workers to a fraction of the available CPUs or the whole count
num_workers = max(1, num_cpus // 2)  # Example: use half of the CPU cores


data_dir = os.path.join('data', dataset)
train_dataset = MemMapDataset(os.path.join(data_dir, 'train.bin'), block_size)
val_dataset = MemMapDataset(os.path.join(data_dir, 'val.bin'), block_size)

from torch.utils.data.distributed import DistributedSampler
from accelerate import DistributedType

# Create a DistributedSampler for the training dataset if in a distributed environment
train_sampler = DistributedSampler(train_dataset, shuffle=True) if accelerator.state.distributed_type != DistributedType.NO else None
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True, num_workers=0)

# Similarly, for the validation dataset
val_sampler = DistributedSampler(val_dataset, shuffle=False) if accelerator.state.distributed_type != DistributedType.NO else None
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, pin_memory=True, num_workers=0)

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = float('inf')

if init_from == 'scratch':
    log("Initializing a new model from scratch")
    model_args = {
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'block_size': block_size,
        'bias': bias,
        'vocab_size': 32064,
        'dropout': dropout
    }
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    log(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"Unknown init_from option: {init_from}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)

model = accelerator.prepare(model)

# -----------------------------------------------------------------------------
# Optimizer Initialization
# -----------------------------------------------------------------------------
optimizer = GPT.configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), 'cuda')
optimizer = accelerator.prepare(optimizer)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def estimate_loss(loader):
    losses = []
    model.eval()
    for X, Y in loader:
        X, Y = accelerator.prepare(X, Y)
        with torch.no_grad():
            _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

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
t0 = time.time()

for epoch in range(max_iters):
    for i, (X, Y) in enumerate(train_loader):
        print("debug enumerate(train_loader) ")
        X, Y = accelerator.prepare(X, Y)
        optimizer.zero_grad()

        print("optimizer.zero_grad()")

        for micro_step in range(gradient_accumulation_steps):
            print("micro_step in range(gradient_accumulation_steps)")

            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if i % log_interval == 0 and accelerator.is_main_process:
                log(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}, Time: {time.time() - t0}")
                t0 = time.time()

            if i % eval_interval == 0 and accelerator.is_main_process:
                train_loss = estimate_loss(train_loader)
                val_loss = estimate_loss(val_loader)
                log(f"Validation - Epoch {epoch}, Iteration {i}, Train Loss: {train_loss}, Val Loss: {val_loss}")
                if val_loss < best_val_loss or always_save_checkpoint:
                    best_val_loss = val_loss
                    checkpoint = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    checkpoint_path = os.path.join(out_dir, f'ckpt_{epoch}_{i}.pt')
                    accelerator.save(checkpoint, checkpoint_path)
                    log(f"Checkpoint saved to {checkpoint_path}")

    iter_num += 1
