import pandas as pd
from datasets import Dataset, get_dataset_config_names, load_dataset


def get_dataset(domain: str = "electronics") -> Dataset:
    """
    https://huggingface.co/datasets/subjqa
    Get the subjqa dataset

    Args:
        domain (str, optional): The domain to load the dataset. Defaults to "electronics".

    Returns:
        Dataset: The subjqa dataset
    """
    domains = get_dataset_config_names("subjqa")
    print(f"List of all domains in `Subjqa` dataset: {domains}\n")

    subjqa = load_dataset(path="subjqa", name=domain)

    return subjqa


def convert_dataset_to_dataframe(dataset: Dataset) -> dict[str, pd.DataFrame]:
    """Get the train/validation/test dataset from subjqa dataset

    Args:
        dataset (Dataset): The subjqa dataset

    Returns:
        dict[str, pd.DataFrame]: train/test/validation -> corresponding dataframe
    """

    dfs = {split: dset.to_pandas() for split, dset in dataset.flatten().items()}
    return dfs
