import re
from typing import Any

from meshy.dataset.base import Dataset
from meshy.utils.sample import Sample, SampleBuilder


def _extract_gsm8k_answer(response: str) -> float | None:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", response)
    if match is None:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


_NUM = r"-?[\d,]+\.?\d*"


def answer_span(text: str) -> str:
    """Everything after the closing think tag; the whole text when absent."""
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):] if idx >= 0 else text


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace(" ", "").rstrip("."))
    except ValueError:
        return None


def lenient_gsm8k_answer(text: str) -> float | None:
    """Extract the answer from the post-think span.

    Order: last ``\\boxed{N}``, last ``#### N``, last bare number. Currency
    symbols and trailing dots are tolerated; only the first numeric run inside
    a boxed expression is read. The training reward keeps the strict
    ``#### N`` rule; this is the holdout scoring口径 only.
    """
    span = answer_span(text)
    boxes = re.findall(r"\\boxed\s*\{([^{}]*)\}", span)
    if boxes:
        m = re.search(_NUM, boxes[-1])
        if m:
            return _to_float(m.group(0))
    strict = _extract_gsm8k_answer(span)
    if strict is not None:
        return strict
    nums = re.findall(_NUM, span)
    return _to_float(nums[-1]) if nums else None


class GSM8K(Dataset):

    def __init__(self, batch_size: int, split: str = "train", hf_kwargs: dict = {}, seed: int | None = None, **kwargs):
        super().__init__(
            hf_kwargs={
                "path": "openai/gsm8k",
                "name": "main",
                "split": split,
                **hf_kwargs,
            },
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )

    def apply_chat_template(self, data: dict, builder: SampleBuilder) -> Sample:
        sample = builder.build_sample([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data["question"] + " Let's think step by step and output the final answer after \"####\"."},
        ])
        sample.ground_truth = _extract_gsm8k_answer(data["answer"])
        return sample

    @staticmethod
    def reward(sample: Sample):
        predicted = _extract_gsm8k_answer(sample.messages[-1]["content"])
        if predicted is None:
            return 0.0
        else:
            return 1.0 if predicted == sample.ground_truth else 0.0
