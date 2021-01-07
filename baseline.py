import argparse

import torch
import torch.optim as optim
from ignite.metrics import Accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification,
                          BertTokenizerFast)

from dataset import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--cuda", default=0, type=int)
parser.add_argument("--dataset", default="chnsenticorp", type=str)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--num_iter", default=10, type=int)
parser.add_argument("--pretrained", default="hfl/chinese-macbert-base", type=str)
args = parser.parse_args()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

config = BertConfig.from_pretrained(args.pretrained)
config.output_hidden_states = True
config.output_attentions = True

tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-macbert-base", config=config)
model = BertForSequenceClassification.from_pretrained(
    args.pretrained, config=config
).to(device)
optimizer = optim.AdamW(
    [
        {"params": model.bert.parameters(), "lr": 1e-6},
        {"params": model.classifier.parameters(), "lr": 1e-5},
    ]
)

dataset = load_dataset()
dataset = dataset.map(
    lambda data: tokenizer(
        data["text"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_token_type_ids=False,
        return_attention_mask=False,
    ),
    batched=True,
    batch_size=8192,
)
dataset.set_format(type="torch", columns=["input_ids", "label"])
dataloader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=12
)

accuracy = Accuracy()
best_acc = 0.0

for iter in range(args.num_iter):
    accuracy.reset()

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        loss, logits, hidden_states, attentions = model(
            input_ids, labels=labels
        ).values()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy.update((logits, labels))

    acc = accuracy.compute()

    if acc > best_acc:
        best_acc = acc
        model.save_pretrained(
            f"/data/model/hfl/chinese-macbert-base/{args.dataset}_{best_acc:.6f}"
        )

    print(f"Iter: {iter}, Acc: {acc:.4f}, Best Acc: {best_acc:.4f}")
