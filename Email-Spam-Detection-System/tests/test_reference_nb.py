import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from spam_detection.reference_nb import ReferenceMultinomialNB


def test_reference_nb_uses_priors_and_smoothing() -> None:
    model = ReferenceMultinomialNB(alpha=1.0).fit(
        [["free", "free"], ["meeting"], ["meeting", "notes"]], ["spam", "ham", "ham"]
    )
    probabilities = model.predict_proba([["free"], ["meeting"], ["unknown"]])
    assert probabilities[0]["spam"] > probabilities[0]["ham"]
    assert probabilities[1]["ham"] > probabilities[1]["spam"]
    assert math.isclose(sum(probabilities[2].values()), 1.0)


def test_reference_matches_manual_posterior() -> None:
    model = ReferenceMultinomialNB(alpha=1.0).fit([["free"], ["meeting"]], ["spam", "ham"])
    result = model.predict_proba([["free"]])[0]
    # vocabulary={free, meeting}; both priors are .5, free|spam=2/3 and free|ham=1/3.
    assert math.isclose(result["spam"], 2 / 3, rel_tol=1e-9)


def test_reference_matches_sklearn_on_deterministic_counts() -> None:
    documents = [["free", "free"], ["free", "offer"], ["meeting"], ["meeting", "notes"]]
    labels = ["spam", "spam", "ham", "ham"]
    reference = ReferenceMultinomialNB().fit(documents, labels)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    counts = vectorizer.fit_transform([" ".join(document) for document in documents])
    sklearn_model = MultinomialNB(alpha=1.0).fit(counts, labels)
    query = vectorizer.transform(["free meeting"])
    expected = dict(zip(sklearn_model.classes_, sklearn_model.predict_proba(query)[0], strict=True))
    actual = reference.predict_proba([["free", "meeting"]])[0]
    for label in expected:
        assert math.isclose(actual[label], expected[label], rel_tol=1e-9)
