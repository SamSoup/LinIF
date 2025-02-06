import copy
from typing import Tuple
import torch
import re
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from kronfluence import Analyzer
from dotenv import load_dotenv
import os

load_dotenv()
CACHE_DIR = os.getenv("CACHE_DIR")


def get_module_summary(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
) -> str:
    model, _ = construct_llm(model_name)
    return Analyzer.get_module_summary(model)


def get_linear_layer_names(summary: str) -> list:
    """
    Extracts the names of all layers of type `Linear` from a module summary.

    Args:
        summary (str): The output string from `Analyzer.get_module_summary(model)`.

    Returns:
        list: A list of strings representing the names of all `Linear` layers.
    """
    linear_layers = []
    # Use regex to match module names where the module type is `Linear`
    pattern = r"Module Name: `(.+?)`, Module: Linear\("
    matches = re.finditer(pattern, summary)
    for match in matches:
        linear_layers.append(match.group(1))

    # remove lm.head, if any
    if "lm_head" in linear_layers:
        linear_layers.remove("lm_head")
    return linear_layers


def construct_llm(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
) -> Tuple[nn.Module, AutoTokenizer]:
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        force_download=True,  # Force clean download for consistency
    )
    return model, tokenizer
