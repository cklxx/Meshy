""""Hendrycks MATH competition dataset, \\boxed answer extraction, and equivalence reward.

Mirrors :mod:`meshy.dataset.gsm8k`'s interface (``apply_chat_template`` +
``reward``) so the same trainer/rollout code drives it. The model is asked to
put its final answer in ``\\boxed{}``; scoring reads only the post-thinking
span (after ``</think>``), takes the **last** boxed expression, and judges
equivalence with ``math_verify`` (SymPy). When ``math_verify`` is absent or
fails to parse either side, it falls back to a canonical-string / numeric
comparison.

Training source: Hendrycks MATH 7.5k train split
(``EleutherAI/hendrycks_math``, seven subject configs concatenated, 7500 rows).
Eval source: ``HuggingFaceH4/MATH-500`` (500 rows, all drawn from MATH test).
"""

from __future__ import annotations

import re
from typing import Any

from meshy.dataset.base import Dataset
from meshy.utils.sample import Sample, SampleBuilder

SYSTEM_PROMPT = "You are a helpful assistant."
SUFFIX = (
    " Let's think step by step and put the final answer in \\boxed{}."
)

# Seven Hendrycks MATH subjects, in the EleutherAI mirror's config naming.
_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def answer_span(text: str) -> str:
    """Everything after the closing think tag; the whole text when absent."""
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):] if idx >= 0 else text


def _matching_brace(text: str, open_idx: int) -> int | None:
    """Index just past the '}' matching the '{' at ``open_idx``; None if unbalanced."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_boxed(text: str) -> str | None:
    """Content of the last balanced ``\\boxed{...}`` in the post-think span.

    Handles nested braces (e.g. ``\\boxed{\\frac{a}{b}}``). A bare
    ``\\boxed x`` (no braces) returns ``x``. Returns None when no boxed answer
    is present.
    """
    span = answer_span(text)
    last = None
    for m in re.finditer(r"\\boxed\b", span):
        i = m.end()
        while i < len(span) and span[i].isspace():
            i += 1
        if i < len(span) and span[i] == "{":
            end = _matching_brace(span, i)
            if end is not None:
                last = span[i + 1:end - 1]
        elif i < len(span):
            # \boxed x shorthand: a single token / command.
            mm = re.match(r"(\\[A-Za-z]+|[^\s\\$}])", span[i:])
            if mm:
                last = mm.group(1)
    return last


def _normalize_for_parse(s: str) -> str:
    """Expand shorthand/variant LaTeX so math_verify can parse it.

    math_verify's parser accepts ``\\frac{1}{2}`` but rejects the shorthand
    ``\\dfrac12`` and the display/text prefixes. Normalise those without
    otherwise changing the expression.
    """
    s = s.strip()
    # no \b here: in "\dfrac12" a digit directly follows "frac", so a word
    # boundary would never match.
    s = re.sub(r"\\[dbt]?frac", r"\\frac", s)
    s = re.sub(r"\\(?:left|right|displaystyle|textstyle)", "", s)

    def _expand(m: re.Match) -> str:
        cmd, a, b = m.group(1), m.group(2), m.group(3)
        if not a.startswith("{"):
            a = "{" + a + "}"
        if not b.startswith("{"):
            b = "{" + b + "}"
        return f"\\{cmd}{a}{b}"

    pat = re.compile(
        r"\\(frac|binom)\s*([^\s{}\\]|\{[^{}]*\})\s*"
        r"([^\s{}\\]|\{[^{}]*\})"
    )
    prev = None
    while prev != s:
        prev = s
        s = pat.sub(_expand, s)
    return s


def _canonicalize_latex(s: str) -> str:
    """Whitespace/command-normalised form for the string fallback."""
    s = _normalize_for_parse(s)
    for cmd in (r"\,", r"\;", r"\:", r"\!", r"\ "):
        s = s.replace(cmd, " ")
    s = s.replace("~", " ")
    return re.sub(r"\s+", "", s)


def _sympy_equiv(p: str, g: str) -> bool:
    """Latex->SymPy exact/simplified-equality; False unless it parses both."""
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex

        return bool(simplify(parse_latex(p) - parse_latex(g)) == 0)
    except Exception:
        return False


def _math_verify_pair(pred: str, gold: str, normalize: bool) -> bool:
    """One math_verify call; optionally parse normalised LaTeX. Never raises."""
    try:
        from math_verify import parse, verify

        g = _normalize_for_parse(gold) if normalize else gold
        p = _normalize_for_parse(pred) if normalize else pred
        return bool(verify(parse(g), parse(p)))
    except Exception:
        return False


def math_equiv(pred: str | None, gold: str | None) -> bool:
    """Judge a boxed prediction against the gold MATH answer.

    Order: exact string; math_verify on raw LaTeX; math_verify again after
    normalising shorthand variants (``\\dfrac12`` → ``\\frac{1}{2}``); a
    canonical-string compare; finally SymPy's LaTeX parser. math_verify (the
    primary judge) and SymPy never raise, so an unparseable answer scores 0.
    """
    if pred is None or gold is None:
        return False
    p, g = pred.strip(), gold.strip()
    if p == g:
        return True
    if _math_verify_pair(p, g, normalize=False):
        return True
    if _math_verify_pair(p, g, normalize=True):
        return True
    if _canonicalize_latex(p) == _canonicalize_latex(g):
        return True
    if _sympy_equiv(_canonicalize_latex(p), _canonicalize_latex(g)):
        return True
    return False


def score_math_response(text: str, gold: str) -> bool:
    """Reward a model response: last post-think \\boxed answer vs gold."""
    return math_equiv(extract_boxed(text), gold)


def strict_math_match(pred: str | None, gold: str | None) -> bool:
    """Strict口径: canonicalised-string equality only (no semantic equivalence).

    The lenient口径 (:func:`math_equiv`) treats ``\\frac12``, ``0.5`` and
    ``1/2`` as equal via SymPy; strict reports how often the emitted answer is
    already in the exact normalised form of the gold (a format-conformance
    number, like eval_gsm8k's strict vs lenient pair).
    """
    if pred is None or gold is None:
        return False
    return _canonicalize_latex(pred) == _canonicalize_latex(gold)


def score_math_response_strict(text: str, gold: str) -> bool:
    return strict_math_match(extract_boxed(text), gold)


class GoldAnswerError(ValueError):
    """A MATH row has no usable boxed gold answer.

    Raised (instead of silently scoring every response 0) when
    ``strict_gold=True``; by default such rows are filtered out with a logged
    warning so they can never become false-negative zero-reward samples that
    skew DAPO's per-group zero-variance drop statistics.
    """


def _row_gold(data: dict[str, Any]) -> str | None:
    if "answer" in data:
        g = data["answer"]
    else:
        g = extract_boxed(data.get("solution", ""))
    return g if isinstance(g, str) and g.strip() else None


def _drop_rows_without_gold(ds, *, strict: bool):
    """Remove (or loudly reject) rows whose boxed gold is empty/None.

    The EleutherAI mirror has two Number Theory rows whose final
    ``\\boxed{}`` is an empty placeholder (the prose answer is 0 but the 0 was
    never put in the box). An empty gold cannot be judged: training on it
    would mark every response wrong regardless of content.
    """
    bad = [i for i, r in enumerate(ds) if _row_gold(r) is None]
    if not bad:
        return ds
    msg = (f"Hendrycks MATH: {len(bad)} row(s) have an empty boxed gold and "
           f"cannot be scored (ids {bad[:10]}{'...' if len(bad) > 10 else ''}); "
           "their prose answer is typically 0 but the box is empty.")
    if strict:
        raise GoldAnswerError(msg + " Set strict_gold=False to filter them out.")
    import logging

    logging.getLogger("meshy.dataset.hendrycks_math").warning(
        msg + " Filtering them out of training so they are not false 0-reward rows."
    )
    keep = [i for i in range(len(ds)) if i not in set(bad)]
    return ds.select(keep)


class HendrycksMATH(Dataset):
    """Hendrycks MATH (EleutherAI mirror, seven subject configs concatenated).

    The base loader binds one config; algebra satisfies its constructor, then
    the underlying dataset is swapped for the concatenation of all seven
    subject splits (7500 train / 5000 test, minus rows with an empty boxed
    gold). The extra algebra load is one cached 1744-row read.
    """

    def __init__(self, batch_size: int, split: str = "train",
                 hf_kwargs: dict = {}, seed: int | None = None,
                 strict_gold: bool = False, **kw):
        super().__init__(
            hf_kwargs={
                "path": "EleutherAI/hendrycks_math",
                "name": "algebra",
                "split": split,
                **hf_kwargs,
            },
            batch_size=batch_size,
            seed=seed,
            **kw,
        )
        full = _load_all_subjects(split)
        self._dropped_gold_rows = [
            i for i, r in enumerate(full) if _row_gold(r) is None
        ]
        self._base_dataset = _drop_rows_without_gold(full, strict=strict_gold)
        self.dataset = self._base_dataset
        if seed is not None:
            self._apply_epoch(0)

    def apply_chat_template(self, data: dict[str, Any], builder: SampleBuilder) -> Sample:
        sample = builder.build_sample([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data["problem"] + SUFFIX},
        ])
        sample.ground_truth = _row_gold(data)
        return sample

    @staticmethod
    def reward(sample: Sample) -> float:
        if not sample.ground_truth:
            # Defensive: a missing gold must never be a 0-reward data point.
            raise GoldAnswerError(
                "empty ground_truth reached the reward; filter upstream "
                "(HendrycksMATH drops these by default)"
            )
        return 1.0 if score_math_response(sample.messages[-1]["content"],
                                         sample.ground_truth) else 0.0


def _load_all_subjects(split: str):
    """Concatenate the seven per-subject splits into one Dataset."""
    from datasets import concatenate_datasets, load_dataset

    parts = [load_dataset("EleutherAI/hendrycks_math", s, split=split)
             for s in _SUBJECTS]
    return concatenate_datasets(parts)
