import pandas as pd
import six
import re
import logging
import json
import jieba
from pathlib import Path
from datasets import Dataset, DatasetInfo


def load_dataset(dataset: str = "ChnSentiCorp", split: str = "train"):
    df = pd.read_csv(f"/data/{dataset}_{split}.tsv", sep="\t")
    ds = Dataset.from_pandas(df)
    ds.features["label"].num_classes = 2
    ds.features["label"].names = ["pos", "neg"]

    return ds



data_folder = Path("datasets/")
LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def tokenize_zh(string):
    string = re.sub(r'\(|\)', '', string)
    return [c.strip() for c in string.strip()]

def load_ocnliDataset(dataset: str = "OCNLI", split: str = "train"):
    print('Loading data: %s' % (dataset),end="\r")
    data = {"premise":[],"hypothesis":[],"label":[]}
    total = 0
    collected = 0
    if split == "train":
        data_file = "train.50k"
    else:
        data_file = split

    with open(f"{data_folder}/{dataset}/{data_file}.json","r") as f:
        for k,line in enumerate(f):
            print(f'Loading data: {dataset}:{k}',end="\r")
            loaded_example = json.loads(line)
            total += 1
            if "gold_label" not in loaded_example:
                loaded_example["gold_label"] = loaded_example["label"]
            if loaded_example["gold_label"] not in LABEL_MAP: continue
            if "sentence1_binary_parse" not in loaded_example:
                sentence1 = tokenize_zh(convert_to_unicode(loaded_example['sentence1']))
                sentence2 = tokenize_zh(convert_to_unicode(loaded_example['sentence2']))
                data["premise"].append(" ".join(jieba.cut(''.join(sentence1))))
                data["hypothesis"].append(" ".join(jieba.cut(''.join(sentence2))))
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            data['label'].append(loaded_example["label"])
            ## log the first example jsut to sanity check
            collected += 1
            # testing
            if total == 300:
                break
            
    df = pd.DataFrame (data, columns = ['premise','hypothesis','label'])
    ds = Dataset.from_pandas(df)
    ds.features["label"].num_classes = 3
    ds.features["label"].names = ["entailment", "neutral","contradiction"]

    return ds
def dataset_for_training(train_dataset,eval_dataset):
    train_text, train_labels = prepare_dataset_for_training(train_dataset)
    eval_text, eval_labels = prepare_dataset_for_training(eval_dataset)
    return train_text,train_labels,eval_text,eval_labels

def prepare_dataset_for_training(datasets_dataset):
    """Changes an `datasets` dataset into the proper format for
    tokenization."""

    def prepare_example_dict(ex):
        """Returns the values in order corresponding to the data.

        ex:
            'Some text input'
        or in the case of multi-sequence inputs:
            ('The premise', 'the hypothesis',)
        etc.
        """
        values = list(ex.values())
        if len(values) == 1:
            return values[0]
        return tuple(values)

    text, outputs = zip(*((prepare_example_dict(x[0]), x[1]) for x in datasets_dataset))
    return list(text), list(outputs)

