# Datasets

This folder mainly contains utility functions to load and preprocess different datasets to run influnce function over.

We generally expect an input dataset to conform to the following format:

```
[
    {
        "prompt": ...
        "completion": ...
    },
    ...
]
```

For huggingface datasets, we preprocess in corresponding `.py` files. The
`utils.py` contains the main tokenization logic.

Mapping:
* test_dataset.py -> simple 10 queries
* sva_dataset.py -> subject verb agreement datasets