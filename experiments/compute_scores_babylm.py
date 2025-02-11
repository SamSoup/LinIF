"""
Note: because this file requires some function in the module, you must call the
function like follows:

For other layers, change to "fully_connected_only". Update `scores_name`
to reflect as well.

We expect f"{output_dir}/{analysis_name}/factors_{factors_name}" to contain
the pre-fitted factors

Will output scores to f"{output_dir}/{analysis_name}/scores_{scores_name}"
"""

import argparse
import logging
from datetime import timedelta
from kronfluence import ScoreArguments
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import default_data_collator
from DAForLinGen.data.babylm_dataset import load_babylm_aochildes
from DAForLinGen.data.utils import load_completion_dataset_generic
from DAForLinGen.models.babylm import get_modules_for_babylm
from DAForLinGen.models.utils import construct_llm
from .task import (
    LanguageModelingTask,
    LanguageModelingWithMarginMeasurementTask,
    LanguageModelingWithContrastTokenTask,
)
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.score_arguments import (
    extreme_reduce_memory_score_arguments,
)
from typing import List
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Influence score computation on Openwebtext dataset."
    )

    # Task Arguments
    parser.add_argument(
        "--task_type",
        type=str,
        default="lm",
        help="One of 'lm' for language model, 'margin', or 'lm_with_contrast_token'"
        "see task.py for the different available tasks",
    )
    parser.add_argument(
        "--contrast_token",
        type=str,
        default=None,
        help="Specific for 'lm_with_contrast_token', specify the token with which"
        "we like to compute the difference between the correct completion w.r.t"
        "the specified contrast token (e.g., 'is' versus 'are')",
    )

    # Model Arguments
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to run influence for",
    )
    parser.add_argument(
        "--module_type",
        type=str,
        help="Specific to Babylm. Specify if `all`, `attention_only`, or"
        " `fully_connected_only` layers are obtained to compute influence",
    )

    # Data Arguments
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/scratch/06782/ysu707/babylm_data/babylm_100M/aochildes.train",
        help="The dataset path to the pre-training data for Babylm.",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=f"/work/06782/ysu707/DAForLinGen/data/datasets/subject_verb_agreement/output.json",
        help="The dataset path to the query data with completion for Babylm.",
    )
    parser.add_argument(
        "--train_start_index",
        type=int,
        default=0,
        help="The starting index for selecting the pre-training dataset subset.",
    )
    parser.add_argument(
        "--train_stop_index",
        type=int,
        default=None,
        help="The stopping index for selecting the pre-training dataset subset. If None, includes all data from start_index.",
    )
    parser.add_argument(
        "--eval_start_index",
        type=int,
        default=0,
        help="The starting index for selecting the evaluation dataset subset.",
    )
    parser.add_argument(
        "--eval_stop_index",
        type=int,
        default=None,
        help="The stopping index for selecting the evaluation dataset subset. If None, includes all data from start_index.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max length of tokens to pad sequences to",
    )

    # Directory Arguments
    parser.add_argument(
        "--analysis_name",
        type=str,
        required=True,
        help="Name of the highest level directory to save results to.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/06782/ysu707",
        help="Directory of output -- will create folder named `analysis_name` at this dir.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        default=False,
        help="Boolean flag to overwrite output directory",
    )

    # Factor/Score Arguments
    parser.add_argument(
        "--factors_name",
        type=str,
        required=True,
        help="Name of the pre-computed factors directory",
    )
    parser.add_argument(
        "--scores_name",
        type=str,
        required=True,
        help="Name of the score directory to save the scores.",
    )
    parser.add_argument(
        "--compute_per_module_scores",
        action="store_true",
        default=False,
        help="Boolean flag to compute influence scores per module."
        "When False -- all modules must have the same factor shape",
    )
    parser.add_argument(
        "--compute_per_token_scores",
        action="store_true",
        default=False,
        help="Boolean flag to compute influence scores per token."
        "When False -- all modules must have the same factor shape",
    )

    # Batch-hyperparameters
    parser.add_argument(
        "--do_extreme_memory_save_for_llms",
        action="store_true",
        help="Boolean flag to for use hyper-params to partition factor compute",
    )
    parser.add_argument(
        "--module_partitions",
        type=int,
        default=4,
        help="Module partitions.",
    )
    parser.add_argument(
        "--data_partitions",
        type=int,
        default=10,
        help="Data partitions.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--query_gradient_accumulation_steps",
        type=int,
        default=10,
        help="Query gradient accumulation_steps.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--per_device_query_batch_size",
        type=int,
        default=1,
        help="Per Device query batch size.",
    )

    # MISC
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

    if args.task_type == "lm_with_contrast_token":
        if args.contrast_token is None:
            raise ValueError(
                "When doing language modeling with contrast token",
                "the contrast token must be specified",
            )
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Initializing Accelerator and distributed environment.")
    kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=5400)
    )  # 1.5 hours.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(
        f"Accelerator initialized. Rank: {accelerator.process_index}, World Size: {accelerator.num_processes}"
    )

    # Synchronize before model preparation
    if accelerator.is_local_main_process:
        logger.info("Main process starting model preparation.")
    else:
        logger.info(
            f"Rank {accelerator.process_index}: Waiting for model preparation."
        )
    accelerator.wait_for_everyone()

    ##### NOTE: Model can also be fairly verbose, change this section as needed
    logger.info("Constructing the model.")
    model, tokenizer = construct_llm(args.model_name)
    summary = Analyzer.get_module_summary(model)

    module_names = get_modules_for_babylm(args.module_type)
    # module_names = get_modules_for_baby_lm_attention()
    get_module_names_fct = lambda: module_names
    logger.info(summary)
    logger.info("Model constructed.")
    logger.info("Linear Layers for Influence Computation.")
    logger.info(module_names)
    #####

    ##### NOTE: Select different task here, depending on desired behavior
    logger.info("Using LanguageModelingTask.")
    if args.task_type == "lm":
        task = LanguageModelingTask(module_fct=get_module_names_fct)
    elif args.task_type == "margin":
        task = LanguageModelingWithMarginMeasurementTask(
            module_fct=get_module_names_fct
        )
    elif args.task_type == "lm_with_contrast_token":
        task = LanguageModelingWithContrastTokenTask(
            module_fct=get_module_names_fct,
            # first input id is always <start_of_seq>, so take 2nd token id
            contrast_token_id=tokenizer(args.contrast_token)["input_ids"][1],
        )

    #####

    model = prepare_model(model, task)
    logger.info("Model preparation completed.")

    # Synchronize before dataset preparation
    if accelerator.is_local_main_process:
        logger.info("Main process starting dataset preparation.")
    else:
        logger.info(
            f"Rank {accelerator.process_index}: Waiting for dataset preparation."
        )
    accelerator.wait_for_everyone()

    # Prepare the datasets
    logger.info("Preparing dataset.")

    def compute_indices(start_index, stop_index) -> List[int]:
        if stop_index is not None and stop_index <= start_index:
            raise ValueError(
                f"Invalid range: stop_index ({stop_index}) must be greater than start_index ({start_index})."
            )
        return (
            list(range(start_index, stop_index))
            if stop_index is not None
            else None
        )

    ##### NOTE: Dataset can be fairly verbose, change this section as needed
    train_dataset = load_babylm_aochildes(
        tokenizer=tokenizer,
        dataset_path=args.train_dataset_path,
        max_length=args.max_length,
        indices=compute_indices(args.train_start_index, args.train_stop_index),
    )

    eval_dataset = load_completion_dataset_generic(
        dataset_path=args.eval_dataset_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        indices=compute_indices(args.eval_start_index, args.eval_stop_index),
    )

    data_collator = default_data_collator

    logger.info(train_dataset)
    logger.info(eval_dataset)
    #####

    logger.info("Dataset preparation completed.")

    # Use Accelerator to prepare the model for distributed training
    logger.info(
        "Preparing the model for distributed training using Accelerator."
    )
    model = accelerator.prepare_model(model)
    logger.info("Model prepared for distributed training.")

    # Analyze pairwise scores
    logger.info("Initializing Analyzer for pairwise score computation.")
    analyzer = Analyzer(
        analysis_name=args.analysis_name,
        model=model,
        task=task,
        profile=args.profile,
        output_dir=args.output_dir,
    )

    # Configure parameters for DataLoader
    logger.info("Configuring DataLoader parameters.")
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4, collate_fn=data_collator, pin_memory=True
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Configure score arguments
    logger.info("Configuring score arguments.")
    print(args.do_extreme_memory_save_for_llms)
    if args.do_extreme_memory_save_for_llms:
        rank = (
            args.query_gradient_rank if args.query_gradient_rank != -1 else None
        )
        score_args = extreme_reduce_memory_score_arguments(
            damping_factor=None,
            module_partitions=args.module_partitions,
            query_gradient_low_rank=rank,
            dtype=torch.bfloat16,
        )
        score_args.query_gradient_accumulation_steps = (
            args.query_gradient_accumulation_steps
        )
        score_args.use_full_svd = True
        score_args.precondition_dtype = torch.float32
        score_args.per_sample_gradient_dtype = torch.float32
        score_args.data_partitions = args.data_partitions
    else:
        score_args = ScoreArguments(use_full_svd=True)
    score_args.compute_per_module_scores = args.compute_per_module_scores
    score_args.compute_per_token_scores = args.compute_per_token_scores

    # Compute pairwise scores
    logger.info("Computing pairwise scores.")
    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        score_args=score_args,
        factors_name=args.factors_name,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=args.per_device_query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    logger.info("Pairwise score computation completed.")

    # Load and log scores
    scores = analyzer.load_pairwise_scores(args.scores_name)
    for split in scores:
        module_scores = scores[split]
        logger.info(f"Scores shape: {module_scores.shape}")


if __name__ == "__main__":
    main()
