import os
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from typing import List


class Word2VecModel:
    def __init__(
        self, model_path=None, vector_size=100, window=5, min_count=5, workers=4
    ):
        self.model_path = model_path
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.stop_words = set(stopwords.words("english"))
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def preprocess_text(self, text: str) -> List[str]:
        text = re.sub(r"\W+", " ", text.lower())
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens

    def train_model(self, sentences: List[str]):
        tokenized_sentences = [self.preprocess_text(sentence) for sentence in sentences]
        self.model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )

    def save_model(self, path: str):
        if self.model:
            self.model.save(path)

    def load_model(self, path: str):
        self.model = Word2Vec.load(path)

    def vectorize_text(self, text: str) -> np.ndarray:
        tokens = self.preprocess_text(text)
        vector = np.mean(
            [self.model.wv[token] for token in tokens if token in self.model.wv], axis=0
        )
        if np.isnan(vector).any():
            return np.zeros(self.vector_size)
        return vector

    def reduce_dimensionality(self, vectors: np.ndarray, n_components=2) -> np.ndarray:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(vectors)

    def get_similarity(self, text1: str, text2: str) -> float:
        vector1 = self.vectorize_text(text1)
        vector2 = self.vectorize_text(text2)
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
