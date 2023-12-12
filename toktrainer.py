from datasets import load_dataset
from tokenizers import AddedToken, SentencePieceBPETokenizer
from multiprocessing import cpu_count

def train_tokenizer(data_files, vocab_size=32_000, min_frequency=2):
    """
    Train a SentencePiece BPE tokenizer on the given dataset.
    """
    # Load dataset with streaming
    dataset = load_dataset('text', data_files=data_files, split='train',num_proc=cpu_count())

    # Initialize tokenizer
    tokenizer = SentencePieceBPETokenizer()

    # Creating a generator
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]


    # Create special tokens to the vocabulary
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "<mask>", "<sos>", "<eos>", "<unk>"]

    varnamala_path = "Parquet/varnamala.txt"
    with open(varnamala_path, 'r', encoding='utf-8') as file:
        varnamala = file.read().split()

    # Train tokenizer
    tokenizer.train_from_iterator( batch_iterator(), 
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens= special_tokens,
        limit_alphabet = 1000,
        initial_alphabet = varnamala,
        length=len(dataset),
        )

    for token in special_tokens:
        tokenizer.add_tokens([AddedToken(token, lstrip=True, rstrip=True)])

    # Save tokenizer
    tokenizer.save("tokenizer.json")

data_files = [
    "Parquet/Gutenberg/*parquet",
    "Parquet/Wikipedia/*parquet",
    "Parquet/AmarUjala/*parquet",
    "Parquet/BookCorpus2/*parquet",
    "Parquet/Dialect/*parquet",
    "Parquet/HackerNews/*parquet",
    "Parquet/NIH_RePORTER/*parquet",
    "Parquet/WikiEn/*parquet",
    "Parquet/YouTube/*parquet"
]

train_tokenizer(data_files)
