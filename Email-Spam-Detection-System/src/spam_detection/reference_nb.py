"""Small educational multinomial Naive Bayes implementation."""

from __future__ import annotations

import math
from collections import Counter


class ReferenceMultinomialNB:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.classes_: list[str] = []
        self.class_log_prior_: dict[str, float] = {}
        self.feature_log_prob_: dict[str, dict[str, float]] = {}
        self.vocabulary_: set[str] = set()

    def fit(self, documents: list[list[str]], labels: list[str]) -> ReferenceMultinomialNB:
        if len(documents) != len(labels) or not labels:
            raise ValueError("documents and labels must have the same non-zero length")
        self.classes_ = sorted(set(labels))
        self.vocabulary_ = set(token for document in documents for token in document)
        if not self.vocabulary_:
            raise ValueError("training vocabulary is empty")
        total = len(labels)
        for cls in self.classes_:
            class_docs = [
                doc for doc, label in zip(documents, labels, strict=False) if label == cls
            ]
            counts = Counter(token for doc in class_docs for token in doc)
            denominator = sum(counts.values()) + self.alpha * len(self.vocabulary_)
            self.class_log_prior_[cls] = math.log(len(class_docs) / total)
            self.feature_log_prob_[cls] = {
                token: math.log((counts[token] + self.alpha) / denominator)
                for token in self.vocabulary_
            }
        return self

    def predict_log_proba(self, documents: list[list[str]]) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for document in documents:
            counts = Counter(token for token in document if token in self.vocabulary_)
            scores = {
                cls: self.class_log_prior_[cls]
                + sum(count * self.feature_log_prob_[cls][token] for token, count in counts.items())
                for cls in self.classes_
            }
            normalizer = max(scores.values()) + math.log(
                sum(math.exp(v - max(scores.values())) for v in scores.values())
            )
            rows.append({cls: score - normalizer for cls, score in scores.items()})
        return rows

    def predict_proba(self, documents: list[list[str]]) -> list[dict[str, float]]:
        return [
            {cls: math.exp(value) for cls, value in row.items()}
            for row in self.predict_log_proba(documents)
        ]
