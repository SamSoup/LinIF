"""
Template File Format

Generally describe how to compute scores given

1) A particular model 

2) A particular pre-training, and query (with LLM completions) dataset

Do NOT Run
"""

"""
Note: because this file requires some function in the module, you must call the
function like follows:

Usage:

cd <main directory>
python -m experiments.compute_scores_babylm \
    ...


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
from DAForLinGen.models.utils import construct_llm, get_linear_layer_names
from .task import (
    LanguageModelingTask,
)
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.score_arguments import (
    extreme_reduce_memory_score_arguments,
)
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Influence score computation on Openwebtext dataset."
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
        default="/scratch/06782/ysu707/babylm_dat",
        help="The dataset path to the pre-training data for Babylm.",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=f"/work/06782/ysu707/DAForLinGen/data/datasets/subject_verb_agreement/output.json",
        help="The dataset path to the query data with completion for Babylm.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="The starting index for selecting the dataset subset.",
    )
    parser.add_argument(
        "--stop_index",
        type=int,
        default=None,
        help="The stopping index for selecting the dataset subset. If None, includes all data from start_index.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max length to pad sequences to",
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

    # Batch-hyperparameters
    parser.add_argument(
        "--do_extreme_memory_save_for_llms",
        action="store_true",
        help="Boolean flag to for use hyper-params to partition factor compute",
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

    module_names = get_linear_layer_names(summary)
    get_module_names_fct = lambda: module_names
    logger.info(summary)
    logger.info("Model constructed.")
    logger.info("Linear Layers for Influence Computation.")
    logger.info(module_names)
    #####

    # Define task and prepare the model
    logger.info("Using LanguageModelingTask.")
    task = LanguageModelingTask(module_fct=get_module_names_fct)
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
    if args.stop_index is not None and args.stop_index <= args.start_index:
        raise ValueError(
            f"Invalid range: stop_index ({args.stop_index}) must be greater than start_index ({args.start_index})."
        )
    indices = (
        list(range(args.start_index, args.stop_index))
        if args.stop_index is not None
        else None
    )
    ##### NOTE: Dataset can be fairly verbose, change this section as needed

    train_dataset = None
    # Should look something like this
    # train_dataset = load_babylm_aochildes(
    #     tokenizer=tokenizer,
    #     dataset_path=args.train_dataset_path,
    #     max_length=args.max_length,
    # )

    eval_dataset = load_completion_dataset_generic(
        dataset_path=args.eval_dataset_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    data_collator = default_data_collator
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
            module_partitions=4,
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
