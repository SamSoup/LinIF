# Attributing Linguistic Phenomena to Training Data

We adapt the [`kronfluence`](https://github.com/pomonam/kronfluence) package
for data attribution.


## Environment Setup 

Conda environment installation on Vista:

```bash
conda create -n influence python=3.10
conda activate influence
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install rbo seaborn datasets transformers peft accelerate langchain-huggingface
python -m pip install kronfluence
python -m pip install scikit-learn jupyter matplotlib tueplots evaluate sentencepiece!=0.1.92 nltk py7zr rouge-score colorama python-dotenv
```

## Geting Started

To perform attribution for a specific model, a pre-training dataset, and a query dataset (with completions), you must make sure the following are complete:

1. Defining some `load_dataset` function in `data/datasets/{dataset}_dataset.py` for both the  pre-training dataset and query dataset. The utility functions in `data/datasets.utils.py` can assist.

2. Define some `load_modules` function in `models/{model}.py` to determine the appropriate names of the linear layers to compute influence over. Note that `kronfluence` only supports linear layers.

A quick way to check all of the model layers is:

```python
from DAForLinGen.models.utils import construct_llm

model, tokenizer = construct_llm(
    model_name=<hf-model-name>
)
print(Analyzer.get_module_summary(model))
```

3. Write a custom `compute_scores.py` and `fit_factors.py` in `experiments`. There are templates where you only have to replace loading the dataset, model, and modules to get started quickly.

4. Optionally, verify that `experiments/task.py` is set up appropriately for the task.

5. The general pipeline is then as follows:
* Run `fit_factors.py`. Will output factors to `{output_dir}/{analysis_name}/f"factors_{factors_name}`
* Optionally, run `inspect_factors.py` to verify factors. Will output quick visualizations of eigenvalues to a desired output folder.
* Run `compute_scores.py`. Will output influence scores to `{output_dir}/{analysis_name}/scores_{scores_name}`. Factors must have been fitted to run this script.
* Optionally, run `inspect_examples.ipynb` with the appropriate scores and dataset loading code to inspect top influential examples
for each query.

## Generating Outputs

To accomodate free-text queries, we provide a `generate.py` to generation responses from a model and save it at a desired location as input to query the model. The expected run command is something like:

```bash
python3 generate.py \
    --dataset_name_or_path "input-csv-path" \
    --model_name_or_path "model" \
    --seed 42 \
    --max_new_tokens 20 \
    --temperature 0.7 \
    --top_k 40 \
    --top_p 0.9 \
    --output_dir "output-path"
```