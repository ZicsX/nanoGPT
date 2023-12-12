import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from multiprocessing import cpu_count
from datasets import concatenate_datasets
from transformers import PreTrainedTokenizerFast

num_proc = cpu_count()
# Initialize the tokenizer

data_files = {
    "AIR": "Parquet/AIR/*.parquet",
    "AmarUjala": "Parquet/AmarUjala/*.parquet",
    "Bhaskar": "Parquet/Bhaskar/*.parquet",
    "Blogging": "Parquet/Blogging/*.parquet",
    "BookCorpus2": "Parquet/BookCorpus2/*.parquet",
    "CC": "Parquet/CC/*.parquet",
    "Dialect": "Parquet/Dialect/*.parquet",
    "Gutenberg": "Parquet/Gutenberg/*.parquet",
    "HackerNews": "Parquet/HackerNews/*.parquet",
    "IndiaTv": "Parquet/IndiaTv/*.parquet",
    "Jagran": "Parquet/Jagran/*.parquet",
    "LiveHindustan": "Parquet/LiveHindustan/*.parquet",
    "Magazines": "Parquet/Magazines/*.parquet",
    "NIH_RePORTER": "Parquet/NIH_RePORTER/*.parquet",
    "Patrika": "Parquet/Patrika/*.parquet",
    "PhilPapers": "Parquet/PhilPapers/*.parquet",
    "WikiEn": "Parquet/WikiEn/*.parquet",
    "Wikipedia": "Parquet/Wikipedia/*.parquet",
    "YouTube": "Parquet/YouTube/*.parquet",
    "mC4Hindi": "Parquet/mC4-Hindi/*.parquet",
}

dataset = load_dataset('parquet',data_files=data_files,columns=['text'])
all_data = concatenate_datasets([dataset[split] for split in dataset.keys()])

# Split the data into train and test sets
train_test_split_ratio = 0.93
split_dataset = all_data.train_test_split(test_size=1 - train_test_split_ratio, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

special_tokens = {
    "pad_token": "[PAD]",
    "bos_token": "[BOS]",  # Beginning of sequence
    "eos_token": "[EOS]",  # End of sequence
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
}
# Initialize the tokenizer
enc = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json", **special_tokens )

def process(example):
    ids = enc.encode(example['text'], add_special_tokens=True)
    ids.append(enc.eos_token_id)
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)
print(sum(tokenized['train']['len']))
print(sum(tokenized['val']['len']))

# # Export single compiled binaries for train and val
# for split, dset in tokenized.items():
#     arr_len = np.sum(dset['len'], dtype=np.uint64)
#     filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
#     dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
#     arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
#     total_batches = 1024

#     idx = 0
#     for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
#         # Batch together samples for faster write
#         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
#         arr_batch = np.concatenate(batch['ids'])
#         # Write into mmap
#         arr[idx : idx + len(arr_batch)] = arr_batch
#         idx += len(arr_batch)
#     arr.flush()

#Export in shards
# concatenate all the ids in each dataset into one large file we can use for training
bytes_per_token = 2  # np.uint16 uses 2 bytes per token
gb_per_shard = 1
tokens_per_shard = gb_per_shard * (1024**3) // bytes_per_token  # number of tokens per 1 GB shard

# Write to sharded binary files
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    shard_index = 0
    shard_size = 0
    shard_filename = os.path.join(os.path.dirname(__file__), f'tokens/{split}_shard{shard_index}.bin')
    dtype = np.uint16
    arr = np.memmap(shard_filename, dtype=dtype, mode='w+', shape=(tokens_per_shard,))
    total_batches = 1024
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {shard_filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])

        # Check if current shard is full
        if shard_size + len(arr_batch) > tokens_per_shard:
            arr.flush()  # Flush current shard
            shard_index += 1  # Move to next shard
            shard_filename = os.path.join(os.path.dirname(__file__), f'tokens/{split}_shard{shard_index}.bin')
            arr = np.memmap(shard_filename, dtype=dtype, mode='w+', shape=(tokens_per_shard,))
            idx = 0  # Reset index for new shard
            shard_size = 0  # Reset shard size

        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
        shard_size += len(arr_batch)

    arr.flush()
