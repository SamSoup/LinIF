import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kronfluence import Analyzer


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Visualize and analyze lambda matrices for all modules."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing .safetensors files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    # Update matplotlib parameters
    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams["axes.axisbelow"] = True

    # Load data from safetensors files
    lambda_processed_path = f"{data_dir}/num_lambda_processed.safetensors"
    lambda_matrix_path = f"{data_dir}/lambda_matrix.safetensors"

    lambda_processed_data = Analyzer.load_file(lambda_processed_path)
    lambda_matrix_data = Analyzer.load_file(lambda_matrix_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all modules in the lambda_processed data
    for module_name in lambda_processed_data.keys():
        print(f"Processing module: {module_name}")

        # Get the data for the current module
        lambda_processed = lambda_processed_data[module_name]
        lambda_matrix = lambda_matrix_data[module_name]

        # Process the lambda matrix
        lambda_matrix.div_(lambda_processed)
        lambda_matrix = lambda_matrix.float()

        # Plot and save lambda matrix
        plt.matshow(lambda_matrix, cmap="PuBu", norm=LogNorm())
        plt.title(module_name)
        plt.colorbar()
        plt.savefig(
            f"{output_dir}/{module_name.replace('.', '_')}_lambda_matrix.png"
        )
        plt.clf()

        # Analyze and save eigenvalues plot
        lambda_matrix = lambda_matrix.view(-1).numpy()
        sorted_lambda_matrix = np.sort(lambda_matrix)
        plt.plot(sorted_lambda_matrix)
        plt.title(module_name)
        plt.grid()
        plt.yscale("log")
        plt.ylabel("Eigenvalues")
        plt.savefig(
            f"{output_dir}/{module_name.replace('.', '_')}_eigenvalues.png"
        )
        plt.clf()

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
