import argparse
import logging
from datetime import timedelta
from transformers import default_data_collator, DataCollatorForLanguageModeling
import torch
from kronfluence.arguments import FactorArguments
from accelerate import Accelerator, InitProcessGroupKwargs
from DAForLinGen.models.utils import construct_llm, get_linear_layer_names
from .task import LanguageModelingTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import (
    extreme_reduce_memory_factor_arguments,
)
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Introduce DDP since vista has 1 gpu per node
import torch.distributed as dist
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Influence factor computation on Openwebtext dataset."
    )

    # Model Arguments
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to run influence for",
    )

    # Data Arguments
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
        help="The stopping index for selecting the dataset subset. "
        "If None, includes all data from start_index.",
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

    # Factor Arguments
    parser.add_argument(
        "--factors_name",
        type=str,
        required=True,
        help="Name of the factor directory to save results to.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=4,
        help="Batch size for computing influence factors.",
    )

    # Batch-hyperparameters
    parser.add_argument(
        "--do_extreme_memory_save_for_llms",
        action="store_true",
        default=False,
        help="Boolean flag to for use hyper-params to partition factor compute",
    )
    parser.add_argument(
        "--covariance_module_partitions",
        type=int,
        default=2,
        help="Number of module partitions for covariance calculations.",
    )
    parser.add_argument(
        "--lambda_module_partitions",
        type=int,
        default=4,
        help="Number of module partitions for lambda calculations.",
    )
    parser.add_argument(
        "--covariance_data_partitions",
        type=int,
        default=4,
        help="Number of data partitions for covariance calculations.",
    )
    parser.add_argument(
        "--lambda_data_partitions",
        type=int,
        default=4,
        help="Number of data partitions for lambda calculations.",
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
        f"Accelerator initialized. Rank: {accelerator.process_index}, "
        "World Size: {accelerator.num_processes}"
    )

    # Synchronize before dataset preparation
    if accelerator.is_local_main_process:
        logger.info("Main process starting dataset preparation.")
    else:
        logger.info(
            f"Rank {accelerator.process_index}: Waiting for dataset preparation."
        )
    accelerator.wait_for_everyone()

    # Prepare the trained model
    logger.info("Constructing the model.")
    model, tokenizer = construct_llm(args.model_name)
    summary = Analyzer.get_module_summary(model)
    module_names = get_linear_layer_names(summary)

    get_module_names_fct = lambda: module_names
    logger.info(summary)
    logger.info("Model constructed.")
    logger.info("Linear Layers for Influence Computation.")
    logger.info(module_names)

    # Define task and prepare model
    logger.info("Preparing the model with the task.")
    task = LanguageModelingTask(get_module_names_fct)
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

    # Prepare the dataset
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
    # Should look like this:
    # train_dataset = load_babylm_aochildes(
    #     tokenizer=tokenizer,
    #     dataset_path=args.train_dataset_path,
    #     max_length=args.max_length,
    # )

    data_collator = default_data_collator
    #####
    logger.info("Dataset preparation completed.")

    # Use Accelerator to prepare the model for distributed training
    logger.info(
        f"Rank {accelerator.process_index}: Preparing model using Accelerator."
    )
    model = accelerator.prepare_model(model)
    logger.info(f"Rank {accelerator.process_index}: Model prepared.")

    # Analyze influence factors
    logger.info("Initializing Analyzer for influence factor computation.")
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
        num_workers=4,
        # collate_fn=default_data_collator,
        # collate_fn=logging_data_collator,
        collate_fn=data_collator,
        pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Configure factor arguments
    logger.info("Configuring factor arguments.")
    if args.do_extreme_memory_save_for_llms:
        factor_args = extreme_reduce_memory_factor_arguments(
            strategy=args.factor_strategy,
            module_partitions=1,
            dtype=torch.bfloat16,
        )
        factor_args.covariance_module_partitions = (
            args.covariance_module_partitions
        )
        factor_args.lambda_module_partitions = args.lambda_module_partitions
        factor_args.covariance_data_partitions = args.covariance_data_partitions
        factor_args.lambda_data_partitions = args.lambda_data_partitions
    else:
        factor_args = FactorArguments(strategy=args.factor_strategy)

    # Fit factors
    logger.info("Fitting influence factors.")
    analyzer.fit_all_factors(
        factors_name=args.factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=args.overwrite_output_dir,
    )
    logger.info("Influence factor computation completed.")


if __name__ == "__main__":
    main()
