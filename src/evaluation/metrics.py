from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

_SPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:final answer|answer|short answer|the answer is|it is|it's)\s*[:\-]\s*",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = value.lower().translate(_PUNCT_TABLE)
    value = _SPACE_RE.sub(" ", value).strip()
    return value


def extract_scoring_answer(prediction: str, references: Iterable[str]) -> str:
    raw = unicodedata.normalize("NFKC", prediction or "").strip()
    if not raw:
        return ""

    # Reference-aware extraction: if the long-form response explicitly contains a gold
    # answer string, score against that short answer instead of the full explanation.
    normalized_raw = normalize_text(raw)
    refs = [str(item).strip() for item in references if str(item).strip()]
    for ref in refs:
        normalized_ref = normalize_text(ref)
        if normalized_ref and normalized_ref in normalized_raw:
            return ref

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in lines:
        stripped = _ANSWER_PREFIX_RE.sub("", line).strip(" -*\t")
        if stripped != line and stripped:
            return stripped

    first_line = lines[0] if lines else raw
    first_sentence = _SENTENCE_SPLIT_RE.split(first_line, maxsplit=1)[0].strip()
    first_sentence = _ANSWER_PREFIX_RE.sub("", first_sentence).strip(" -*\t")
    return first_sentence or first_line


def exact_match(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_text(prediction)
    refs = [normalize_text(item) for item in references if normalize_text(item)]
    if not refs:
        return 0.0
    return 1.0 if pred in refs else 0.0


def answer_contains(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_text(prediction)
    refs = [normalize_text(item) for item in references if normalize_text(item)]
    if not pred or not refs:
        return 0.0
    for ref in refs:
        if ref in pred or pred in ref:
            return 1.0
    return 0.0


def token_f1(prediction: str, references: Iterable[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens_list = [normalize_text(item).split() for item in references if normalize_text(item)]
    if not pred_tokens or not ref_tokens_list:
        return 0.0

    best = 0.0
    pred_counter = Counter(pred_tokens)
    for ref_tokens in ref_tokens_list:
        if not ref_tokens:
            continue
        ref_counter = Counter(ref_tokens)
        common = pred_counter & ref_counter
        overlap = sum(common.values())
        if overlap <= 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        best = max(best, score)
    return best


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def anls(prediction: str, references: Iterable[str], threshold: float = 0.5) -> float:
    pred = normalize_text(prediction)
    refs = [normalize_text(item) for item in references if normalize_text(item)]
    if not pred or not refs:
        return 0.0

    best = 0.0
    for ref in refs:
        distance = _levenshtein_distance(pred, ref)
        norm = distance / max(len(pred), len(ref), 1)
        score = 1.0 - norm
        if score > best:
            best = score
    return best if best >= threshold else 0.0


def hit_at_k(relevant: list[bool]) -> float:
    return 1.0 if any(relevant) else 0.0


def recall_at_k(relevant: list[bool], num_relevant: int = 1) -> float:
    if num_relevant <= 0:
        return 0.0
    return min(sum(1 for flag in relevant if flag) / num_relevant, 1.0)


def precision_at_k(relevant: list[bool]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for flag in relevant if flag) / len(relevant)


def citation_accuracy(first_hit: bool) -> float:
    return 1.0 if first_hit else 0.0
