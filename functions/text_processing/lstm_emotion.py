import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import numpy as np
import joblib
import os


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class LSTMEmotionModel(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, vocab_size, output_dim, n_layers, dropout
    ):
        super(LSTMEmotionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embedded)
        hidden = torch.cat(
            (
                lstm_out[:, -1, : self.lstm.hidden_size],
                lstm_out[:, 0, self.lstm.hidden_size :],
            ),
            dim=1,
        )
        output = self.fc(self.dropout(hidden))
        return output


class EmotionAnalyzer:
    def __init__(self, max_length=100, batch_size=32, epochs=10, learning_rate=1e-3):
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_encoder = LabelEncoder()

    def preprocess(self, texts, labels=None):
        if labels is not None:
            labels = self.label_encoder.fit_transform(labels)
        return texts, labels

    def train(self, texts, labels):
        texts, labels = self.preprocess(texts, labels)
        dataset = TextDataset(texts, labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        vocab_size = self.tokenizer.vocab_size
        model = LSTMEmotionModel(
            embedding_dim=128,
            hidden_dim=256,
            vocab_size=vocab_size,
            output_dim=1,
            n_layers=2,
            dropout=0.3,
        )
        model = model.to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device).float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        self.model = model

    def save_model(self, model_path, tokenizer_path, label_encoder_path):
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.tokenizer, tokenizer_path)
        joblib.dump(self.label_encoder, label_encoder_path)

    def load_model(self, model_path, tokenizer_path, label_encoder_path):
        vocab_size = self.tokenizer.vocab_size
        model = LSTMEmotionModel(
            embedding_dim=128,
            hidden_dim=256,
            vocab_size=vocab_size,
            output_dim=1,
            n_layers=2,
            dropout=0.3,
        )
        model.load_state_dict(torch.load(model_path))
        self.model = model.to(self.device)
        self.tokenizer = joblib.load(tokenizer_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        return torch.sigmoid(outputs).item()
