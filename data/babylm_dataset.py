from transformers import AutoTokenizer
from typing import List, Optional
from torch.utils import data
from datasets import Dataset, load_dataset
from .utils import tokenize_dataset
from dotenv import load_dotenv
import os


def load_babylm_aochildes(
    tokenizer: AutoTokenizer,
    dataset_path="/scratch/06782/ysu707/babylm_data/babylm_100M/aochildes.train",
    max_length: int = 512,
    indices: Optional[List[int]] = None,
) -> Dataset:
    load_dotenv()
    CACHE_DIR = os.getenv("CACHE_DIR")
    splits = {"train": dataset_path}
    datasets = load_dataset("text", data_files=splits, cache_dir=CACHE_DIR)
    raw_dataset = datasets["train"]

    # Subset the dataset if indices are provided
    if indices is not None:
        raw_dataset = raw_dataset.select(indices)

    return tokenize_dataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        indices=indices,
    )
