from pathlib import Path

import pytest

from spam_detection.data import combine_email, load_dataset, normalize_label


def test_html_and_sensitive_markers_are_normalized() -> None:
    text = combine_email("Hi", "<b>Visit https://example.test</b> or mail me@example.test")
    assert "url" in text and "email" in text and "<b>" not in text


def test_labels_are_normalized() -> None:
    assert normalize_label("SPAM") == "spam"
    assert normalize_label("0") == "ham"
    with pytest.raises(ValueError):
        normalize_label("maybe")


def test_fixture_profile_detects_both_classes() -> None:
    path = Path(__file__).parents[1] / "data" / "fixtures" / "toy_emails.csv"
    frame, profile = load_dataset(path)
    assert len(frame) == 12
    assert profile.spam == 6 and profile.ham == 6
