from transformers import AutoTokenizer
from typing import List, Optional
from torch.utils import data
from datasets import Dataset, load_dataset
from .utils import tokenize_dataset
import torch.distributed as dist
from dotenv import load_dotenv
import os

load_dotenv()
CACHE_DIR = os.getenv("WORK_DIR")


def load_openwebtext_dataset(
    tokenizer: AutoTokenizer,
    indices: Optional[List[int]] = None,
    max_length: int = 512,
) -> data.Dataset:

    raw_datasets = load_dataset("Elriggs/openwebtext-100k", cache_dir=CACHE_DIR)
    raw_dataset = raw_datasets["train"]

    # Apply additional filtering if indices are provided
    if indices is not None:
        raw_dataset = raw_dataset.select(indices)

    # Check if distributed is initialized and get rank
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    # Dataset tokenization logic
    return tokenize_dataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        indices=indices,
    )
