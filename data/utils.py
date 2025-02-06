"""
Utility Functions
"""

from typing import Optional, List
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils import data
import torch.distributed as dist
import copy
import json


def load_completion_dataset_generic(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    indices: List[int] = None,
) -> data.Dataset:
    # Load the raw dataset
    with open(dataset_path, "r") as handle:
        data = json.load(handle)

    raw_dataset = Dataset.from_dict(
        {
            "prompt": [item["prompt"] for item in data],
            "completion": [item["completion"] for item in data],
        }
    )

    return tokenize_completion_dataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        indices=indices,
    )


def tokenize_dataset(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    indices: List[int] = None,
) -> data.Dataset:
    """
    Loads a dataset, optionally subsets it by indices, and tokenizes it.
    We do not expect the dataset to contain a completion. Instead, the dataset
    should only contain one column for tokenization.

    Args:
        raw_dataset (Dataset): The raw dataset, loaded from hf or disk.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the dataset.
        max_length (int): Maximum sequence length for tokenization (default: 512).

    Returns:
        datasets.Dataset: The tokenized dataset.
    """
    # Select specific indices if provided
    if indices is not None:
        raw_dataset = raw_dataset.select(indices)

    # Check distributed environment
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    # Set tokenizer pad token, if not already
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine the text column
    column_names = raw_dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Define the tokenization function
    def tokenize_function(examples):
        results = tokenizer(
            examples[text_column_name],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Mask padding tokens for loss computation
        results["labels"] = [
            [
                -100 if token == tokenizer.pad_token_id else token
                for token in label
            ]
            for label in results["input_ids"]
        ]
        return results

    # Tokenization process
    if rank == 0:
        print("Main process: Tokenizing dataset", flush=True)
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Tokenizing dataset",
        )
        if is_distributed:
            dist.barrier()  # Notify other processes
    else:
        print(
            f"Rank {rank}: Waiting for main process to tokenize dataset",
            flush=True,
        )
        dist.barrier()
        tokenized_dataset = (
            raw_dataset  # Assumes dataset was already tokenized by main process
        )

    return tokenized_dataset


def tokenize_completion_dataset(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    indices: Optional[List[int]] = None,
) -> data.Dataset:
    """
    Loads a dataset, optionally subsets it by indices, and tokenizes it.
    The dataset is expected to contain two columns: "prompt" and "completion"

    Args:
        raw_dataset (Dataset): The raw dataset, loaded from hf or disk.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the dataset.
        max_length (int): Maximum sequence length for tokenization (default: 512).

    Returns:
        datasets.Dataset: The tokenized dataset.
    """
    # Subset dataset
    if indices is not None:
        raw_dataset = raw_dataset.select(indices)

    # Check distributed environment
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    # Set tokenizer pad token, if not already
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        data_dict = {}
        prompt_results = tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        completion_results = tokenizer(
            examples["completion"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = (
            prompt_results["input_ids"] + completion_results["input_ids"][1:]
        )
        attention_mask = (
            prompt_results["attention_mask"]
            + completion_results["attention_mask"][1:]
        )
        data_dict["input_ids"] = input_ids
        data_dict["labels"] = copy.deepcopy(input_ids)
        data_dict["labels"][: len(prompt_results["input_ids"])] = [
            -100 for _ in range(len(prompt_results["input_ids"]))
        ]
        data_dict["attention_mask"] = attention_mask
        return data_dict

    # Dataset tokenization logic
    if rank == 0:
        print("Main process: Running tokenizer on dataset", flush=True)
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=False,
            num_proc=None,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        if is_distributed:
            dist.barrier()  # Notify other ranks that tokenization is complete
    else:
        print(
            f"Rank {rank}: Waiting for dataset tokenization to complete",
            flush=True,
        )
        dist.barrier()
        tokenized_dataset = (
            raw_dataset  # Other ranks will access the already processed dataset
        )

    return tokenized_dataset
