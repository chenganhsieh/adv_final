# coding: utf-8
import torch
import torch.nn as nn
from transformers import BertModel


# Bert-BiGRU-Classifier
class BiGRU(nn.Module):
    def __init__(self):
        super(BiGRU, self).__init__()
        self.embedding = BertModel.from_pretrained('hfl/chinese-macbert-base')
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=768,
            dropout=0.3,
            num_layers=5,
            bidirectional=True,
            batch_first=True,
        )

        self.fc_1 = nn.Linear(768*2, 128)
        self.fc_2= nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens1, tokens2,masks1=None,masks2=None):
        # labels
        # self.fc_1 = nn.Linear(768*2, amount_labels)
        # self.fc_2= nn.Linear(128, amount_labels)
        # BERT
        embedded1, _ = self.embedding(tokens1, attention_mask=masks1)
        embedded2, _ = self.embedding(tokens2, attention_mask=masks2)
        cls_vector1 = embedded1[:, 0, :]
        cls_vector1 = cls_vector1.view(-1, 1, 768)

        cls_vector2 = embedded2[:, 0, :]
        cls_vector2 = cls_vector2.view(-1, 1, 768)

        # GRU
        _, hidden1 = self.gru(cls_vector1)
        hidden1 = hidden1[-1]

        _, hidden2 = self.gru(cls_vector2)
        hidden2 = hidden2[-1]

        concat_tensor = torch.cat((hidden1, hidden2), 1)
        
        
        # Fully-connected layer
        outputs = self.fc_1(concat_tensor.squeeze(0))
        outputs = self.fc_2(outputs)
        outputs = self.sigmoid(outputs)
        

        return outputs


# Bert-BiGRU-Classifier
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.embedding = BertModel.from_pretrained('bert-base-chinese')
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=768,
            dropout=0.3,
            num_layers=5,
            bidirectional=True,
            batch_first=True,
        )

        self.fc_1 = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens, masks=None):
        # BERT
        embedded, _ = self.embedding(tokens, attention_mask=masks)
        cls_vector = embedded[:, 0, :]
        cls_vector = cls_vector.view(-1, 1, 768)

        # GRU
        _, hidden = self.lstm(cls_vector)
        hidden = hidden[-1]

        # Fully-connected layer
        outputs = self.fc_1(hidden.squeeze(0))
        outputs = self.sigmoid(outputs).view(-1)

        return outputs