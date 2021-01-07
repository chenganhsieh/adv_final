from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import scipy
import torch
import tqdm
import transformers
import textattack
from textattack.commands import TextAttackCommand
from textattack.commands.attack.attack_args import (
    HUGGINGFACE_DATASET_BY_MODEL,
    TEXTATTACK_DATASET_BY_MODEL,
)
from textattack.commands.attack.attack_args_helpers import (
    add_dataset_args,
    add_model_args,
    parse_dataset_from_args,
    parse_model_from_args,
)
from torch.utils.data import DataLoader, RandomSampler

def batch_encode(tokenizer, text_list):
    if hasattr(tokenizer, "batch_encode"):
        return tokenizer.batch_encode(text_list)
    else:
        return [tokenizer.encode(text_input) for text_input in text_list]

def _make_dataloader(tokenizer, text, labels, batch_size):
    """Create torch DataLoader from list of input text and labels.

    :param tokenizer: Tokenizer to use for this text.
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """
    text_ids = batch_encode(tokenizer, text)
    input_ids = np.array(text_ids)
    labels = np.array(labels)
    data = list((ids, label) for ids, label in zip(input_ids, labels))
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

logger = textattack.shared.logger

model = parse_model_from_args(args)


val_dataset = load_ocnliDataset(split="dev")
val_hugdataset = HuggingFaceDataset(val_dataset)


preds = []
ground_truth_outputs = []
i = 0

while i < args.num_examples:
    dataset_batch = val_hugdataset[
        i : min(args.num_examples, i + args.model_batch_size)
    ]
    batch_inputs = []
    for (text_input, ground_truth_output) in dataset_batch:
        attacked_text = textattack.shared.AttackedText(text_input)
        batch_inputs.append(attacked_text.tokenizer_input)
        ground_truth_outputs.append(ground_truth_output)
    batch_preds = self.get_preds(model, batch_inputs)

    if not isinstance(batch_preds, torch.Tensor):
        batch_preds = torch.Tensor(batch_preds)

    preds.extend(batch_preds)
    i += args.model_batch_size

preds = torch.stack(preds).squeeze().cpu()
ground_truth_outputs = torch.tensor(ground_truth_outputs).cpu()

logger.info(f"Got {len(preds)} predictions.")

if preds.ndim == 1:
    # if preds is just a list of numbers, assume regression for now
    # TODO integrate with `textattack.metrics` package
    pearson_correlation, _ = scipy.stats.pearsonr(ground_truth_outputs, preds)
    spearman_correlation, _ = scipy.stats.spearmanr(ground_truth_outputs, preds)

    logger.info(f"Pearson correlation = {_cb(pearson_correlation)}")
    logger.info(f"Spearman correlation = {_cb(spearman_correlation)}")
else:
    guess_labels = preds.argmax(dim=1)
    successes = (guess_labels == ground_truth_outputs).sum().item()
    perc_accuracy = successes / len(preds) * 100.0
    perc_accuracy = "{:.2f}%".format(perc_accuracy)
    logger.info(f"Correct {successes}/{len(preds)} ({_cb(perc_accuracy)})")

