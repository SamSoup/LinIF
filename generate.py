"""
Loads a generative model and run generations on a provided lsit of texts.
Assumes that the dataset is already preprocessed with prompts.

Note the script depends on an environment variable called
`CACHE_DIR` for caching models

Usage: see python3 main.py --help
"""

import torch
import argparse
import json
import os
import random
from typing import Callable
import pandas as pd
from tqdm import tqdm
from transformers import (
    pipeline,
    set_seed,
)
from dotenv import load_dotenv
from torch.utils.data import Dataset
from pprint import pprint
import numpy as np


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def get_args():
    parser = argparse.ArgumentParser()
    # Either give json, or provide all others
    parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Path to the dataset. Must be a .csv containing the input column as 'text'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to the pretrained model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2,
        help="Maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for randomness.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Sample from the best k (number of) tokens. 0 means off (Default: 0, 0 ≤ top_k < 100000).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Sample from the set of tokens with highest probability such that sum of probabilies is higher than p. Lower values focus on the most probable tokens.Higher values sample more low-probability tokens (Default: 0.9, 0 < top_p ≤ 1)",
    )
    parser.add_argument(
        "--is_classification",
        action="store_true",
        help="If set, process the output as classification.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for classification.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for resultant representations. Creates a .json"
        "contaning a list of dict objects containing 'prompt' and 'completions'",
    )
    args = parser.parse_args()

    # If config argument is provided, load configuration from JSON file
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        # Overwrite args with config
        for key, value in config.items():
            setattr(args, key, value)

    print("Arguments passed:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


def generate_responses(
    pipeline: Callable, dataset: Dataset, args: argparse.Namespace
):
    results = []
    # Iterate over the dataset; It is recommended that we iterate directly
    # over the dataset without needing to batch
    print("*** Running Model Inference ***")
    for output in tqdm(
        pipeline(
            dataset,
            max_new_tokens=args.max_new_tokens,
            do_sample=True if args.temperature > 0 else False,
            num_return_sequences=1,
            temperature=args.temperature if args.temperature > 0 else None,
            top_k=args.top_k,
            top_p=args.top_p,
            return_full_text=False,  # only return added text
        )
    ):
        generated_text = output[0]["generated_text"].strip()
        results.append(generated_text)
    return results


def main():
    args = get_args()

    # Create Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Set Randomness
    seed = args.seed
    set_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    ## Load Dataset, from disk because
    ## datasets might require a GLIBC version higher
    ## than what is available on TACC
    fname: str = args.dataset_name_or_path
    if os.path.isfile(fname) and fname.endswith(".csv"):
        raw_dataset = pd.read_csv(args.dataset_name_or_path)
    else:
        raise ValueError("Invalid Dataset path")

    texts = raw_dataset["text"]
    dataset = SentenceDataset(texts.tolist())

    ### Log the first input to check format
    print("First input:\n", dataset[0])

    # Construct pipeline, and run results
    load_dotenv()  # load cache_dir from .env
    text_generator = pipeline(
        "text-generation",
        model=args.model_name_or_path,
        device_map="auto",
        model_kwargs={"cache_dir": os.getenv("CACHE_DIR")},
    )
    results = generate_responses(text_generator, dataset, args)

    # Serialize results to disk
    if len(texts) != len(results):
        raise ValueError(
            "The lengths of 'texts' and 'outputs' must be the same."
        )

    # Convert to list of dictionaries
    outputs = [{"prompt": t, "completion": o} for t, o in zip(texts, results)]

    # Save to a JSON file
    output_file = os.path.join(args.output_dir, "output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
