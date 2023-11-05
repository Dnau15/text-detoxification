import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict


def preprocess_dataset(path="../data/raw/filtered.tsv"):
    """
    This function preprocess dataset that was initially given

    Args:
        path (str, optional): dataset path. Defaults to ".../data/raw/filtered.tsv".

    Returns:
        _type_: this function returns preprocessed dataframe
    """
    data = pd.read_table(path)
    conditions = [data.ref_tox < data.trn_tox ]
    values = ['true']
    data['swap'] = np.select(conditions, values)
    
    is_swap = data['swap'] == 'true'
    data.loc[is_swap, ['reference', 'translation', 'ref_tox', 'trn_tox']] = (
        data.loc[is_swap, ['translation', 'reference', 'trn_tox', 'ref_tox']].values
        )
    
    index_drop = data[(data['ref_tox'] < 0.9) | (data['trn_tox'] > 0.1) ].index
    data.drop(index_drop, inplace=True)
    data.drop(columns=["swap"], axis=1, inplace=True)
    data.to_csv("../data/raw/filtered_converted.csv")
    return data


def get_dataset(path="../data/raw/filtered_converted.csv", n_rows=20000):
    """
    This function returns DataDict dataset from datasets lib

    Args:
        path (str, optional): path of csv file. Defaults to "../data/raw/filtered_converted.csv".

    Returns:
        _type_: DataDict dataset
    """
    raw_datasets = load_dataset("csv", data_files=path)
    raw_datasets = raw_datasets['train'].train_test_split(test_size=1-n_rows/raw_datasets.num_rows["train"], seed=42)
    raw_datasets_train = raw_datasets['train'].train_test_split(test_size=0.3, seed=42)
    raw_datasets_test = raw_datasets_train['test'].train_test_split(test_size=0.5, seed=42)

    ds_splits = DatasetDict({
        'train': raw_datasets_train['train'],
        'valid': raw_datasets_test['train'],
        'test': raw_datasets_test['test']
    })
    return ds_splits


def preprocess_function_extra(tokenizer, model_checkpoint):
    """
    This function tokenizes sentences 

    Args:
        tokenizer (_type_): tokenizer
        model_checkpoint (_type_): checkpoint of the model

    Returns:
        _type_: _description_
    """
    max_input_length = 128
    max_target_length = 128
    if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "translate tox to detox: "
    else:
        prefix = ""
    def preprocess_function(examples):
        inputs = [prefix + ex for ex in examples["reference"]]
        targets = [ex for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function