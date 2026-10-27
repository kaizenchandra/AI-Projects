"""Dataset validation and safe email text ingestion."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value[:200_000])
    parser.close()
    return " ".join(parser.parts)


def normalize_text(value: str, *, max_chars: int = 200_000) -> str:
    value = unicodedata.normalize("NFKC", str(value))[:max_chars]
    value = html_to_text(value)
    value = re.sub(r"(?i)https?://\S+|www\.\S+", " URL ", value)
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " EMAIL ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def combine_email(subject: Any = "", body: Any = "", *, max_chars: int = 200_000) -> str:
    subject_text = normalize_text(subject, max_chars=max_chars)
    body_text = normalize_text(body, max_chars=max_chars)
    combined = f"subject {subject_text} body {body_text}".strip()
    if len(combined) <= max_chars:
        return combined
    return combined[:max_chars]


def parse_eml(path: str | Path, *, max_bytes: int = 2_000_000) -> tuple[str, str]:
    raw = Path(path).read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(f"raw email exceeds the {max_bytes} byte limit")
    message = BytesParser(policy=policy.default).parsebytes(raw)  # type: ignore[arg-type]
    subject = str(message.get("subject", ""))
    bodies: list[str] = []
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.get_content_maintype() != "text":
            continue
        try:
            content = part.get_content()  # type: ignore[attr-defined]
        except (LookupError, UnicodeError) as exc:
            raise ValueError("email contains unsupported character encoding") from exc
        if isinstance(content, str):
            bodies.append(content)
    return subject, "\n".join(bodies)


def normalize_label(value: Any) -> str:
    label = str(value).strip().lower()
    if label in {"spam", "1", "true", "junk"}:
        return "spam"
    if label in {"ham", "0", "false", "legitimate", "not spam"}:
        return "ham"
    raise ValueError(f"unsupported label {value!r}; expected spam or ham")


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    spam: int
    ham: int
    empty: int
    exact_duplicates: int
    near_duplicate_candidates: int
    fingerprint: str


def load_dataset(
    path: str | Path, *, max_message_chars: int = 200_000
) -> tuple[pd.DataFrame, DatasetProfile]:
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError("dataset must contain a label column")
    if "text" not in frame.columns and not {"subject", "body"}.issubset(frame.columns):
        raise ValueError("dataset must contain text, or both subject and body columns")
    if "text" in frame.columns:
        texts = [combine_email("", value, max_chars=max_message_chars) for value in frame["text"]]
    else:
        texts = [
            combine_email(subject, body, max_chars=max_message_chars)
            for subject, body in zip(frame["subject"], frame["body"], strict=False)
        ]
    labels = [normalize_label(value) for value in frame["label"]]
    result = pd.DataFrame({"text": texts, "label": labels})
    if result.empty:
        raise ValueError("dataset contains no records")
    empty = int((result["text"].str.replace("subject body", "", regex=False).str.len() == 0).sum())
    if empty:
        raise ValueError(f"dataset contains {empty} empty email(s)")
    hashes = result["text"].map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    exact_duplicates = int(hashes.duplicated(keep=False).sum())
    # MinHash/semantic similarity belongs in corpus-specific ingestion; this conservative signal
    # catches repeated templates without adding a dependency to the baseline.
    near_duplicate_candidates = int(
        result["text"].str.replace(r"\d+", "#", regex=True).duplicated(keep=False).sum()
    )
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{label}\t{text}" for text, label in zip(result["text"], result["label"], strict=False)
        ).encode()
    ).hexdigest()
    profile = DatasetProfile(
        rows=len(result),
        spam=int((result["label"] == "spam").sum()),
        ham=int((result["label"] == "ham").sum()),
        empty=empty,
        exact_duplicates=exact_duplicates,
        near_duplicate_candidates=near_duplicate_candidates,
        fingerprint=fingerprint,
    )
    if profile.spam == 0 or profile.ham == 0:
        raise ValueError("dataset must contain both spam and ham labels")
    return result, profile
