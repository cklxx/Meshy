"""CPU-only checks for MATH answer extraction, equivalence judging, pass@k."""

import pytest

from meshy.dataset.hendrycks_math import (
    _canonicalize_latex,
    _normalize_for_parse,
    extract_boxed,
    math_equiv,
    score_math_response,
)
from meshy.utils.passatk import aggregate_pass_at_k, pass_at_k


THINK = "<think>work</think>\n"


@pytest.mark.parametrize("text,want", [
    (THINK + r"\boxed{42}", "42"),
    # nested braces: last balanced boxed
    (THINK + r"a \boxed{\frac{1}{2}} b", r"\frac{1}{2}"),
    # multiple boxed -> last
    (THINK + r"\boxed{1} then \boxed{2}", "2"),
    # nested with \dfrac and brackets inside
    (THINK + r"\boxed{\left(3,\dfrac{\pi}{2}\right)}",
     r"\left(3,\dfrac{\pi}{2}\right)"),
    # interval / set with units
    (THINK + r"\boxed{[1,2]\text{ cm}}", r"[1,2]\text{ cm}"),
    # unbalanced braces -> ignored, returns None
    (THINK + r"\boxed{\frac{1}{2}", None),
    # bare \boxed x shorthand
    (THINK + r"\boxed{x}", "x"),
    # boxed only inside the think block is ignored
    (r"<think>\boxed{99}</think>answer is \boxed{7}", "7"),
    ("no box here", None),
])
def test_extract_boxed(text, want):
    assert extract_boxed(text) == want


def test_extract_boxed_handles_nested_curlies():
    # deeply nested: \boxed{\frac{d}{dx}[x^2]=2x}
    got = extract_boxed(THINK + r"\boxed{\frac{d}{dx}[x^2]=2x}")
    assert got == r"\frac{d}{dx}[x^2]=2x"


@pytest.mark.parametrize("pred,gold", [
    # decimal vs fraction
    ("0.5", r"\frac{1}{2}"),
    ("1/2", r"\frac{1}{2}"),
    # dfrac shorthand vs frac full
    (r"\dfrac12", r"\frac{1}{2}"),
    (r"\frac{1}{2}", r"\dfrac{1}{2}"),
    # spacing/command normalisation without math_verify-style exactness
    (r"\left(3,\frac{\pi}{2}\right)", r"(3,\frac{\pi}{2})"),
    # bare identical string
    ("42", "42"),
])
def test_equiv_true(pred, gold):
    assert math_equiv(pred, gold), f"{pred!r} should equal {gold!r}"


@pytest.mark.parametrize("pred,gold", [
    ("0.5", r"\frac{1}{3}"),
    ("2", r"\frac{1}{2}"),
    ("[1,2]", "[1,3]"),
    (None, "1"),
    ("3.14", r"\pi"),
])
def test_equiv_false(pred, gold):
    assert not math_equiv(pred, gold)


def test_normalize_expands_dfrac_shorthand():
    assert _normalize_for_parse(r"\dfrac12") == r"\frac{1}{2}"
    assert _normalize_for_parse(r"\tfrac{a}{b}") == r"\frac{a}{b}"
    assert r"\frac{1}{2}" == _canonicalize_latex(r" \dfrac {1} {2} ")


def test_score_uses_last_boxed_and_post_think():
    gold = r"\frac{1}{2}"
    ok = THINK + r"so \boxed{0.5}"
    assert score_math_response(ok, gold) == 1.0
    bad = THINK + r"\boxed{0.5}\nfinally \boxed{0.25}"
    assert score_math_response(bad, gold) == 0.0
    # wrong-span boxed in thinking is not scored
    wrong_block = r"<think>\boxed{99}</think>no final box"
    assert score_math_response(wrong_block, "99") == 0.0


# ---- pass@k estimator (Chen et al. 2021, eq. 3) ----

def test_pass_at_k_all_correct():
    assert pass_at_k(4, 4, 1) == 1.0
    assert pass_at_k(4, 4, 4) == 1.0


def test_pass_at_k_none_correct():
    assert pass_at_k(4, 0, 1) == 0.0
    assert pass_at_k(8, 0, 4) == 0.0


def test_pass_at_k_exact_small_examples():
    # n=4, c=1, k=1: 1 - C(3,1)/C(4,1) = 1 - 3/4 = 0.25
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    # n=4, c=1, k=4: any one correct among 4 always included -> 1.0
    assert pass_at_k(4, 1, 4) == 1.0
    # n=5, c=2, k=2: 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)


def test_pass_at_k_monotone_in_k():
    vals = [pass_at_k(16, 4, k) for k in (1, 2, 4, 8, 16)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_pass_at_k_raises_when_k_gt_n():
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 4)


def test_aggregate_reports_only_supported_k():
    # two problems: one with 8 samples, one with 32 -> k<=8 uses both,
    # k=16/32 use only the second problem.
    counts = [(8, 2), (32, 8)]
    agg = aggregate_pass_at_k(counts)
    assert set(agg) == {1, 2, 4, 8, 16, 32}
    assert agg[1]["problems"] == 2
    assert agg[16]["problems"] == 1
    assert agg[16]["samples_per_problem"] == 32
    # means equal the single problem's estimator when one contributes
    assert agg[16]["pass_at_k"] == pytest.approx(pass_at_k(32, 8, 16))
    assert agg[32]["pass_at_k"] == pytest.approx(pass_at_k(32, 8, 32))


# ---- T11c: math_verify 0.9.0 tuple-parsing regression, dataset-free -------
# math_verify alone parses (a,b) to only its last component, so it wrongly
# equates tuples differing in an earlier slot. math_equiv must not.

def test_math_verify_upstream_tuple_bug_is_guarded():
    from math_verify import parse, verify
    # documents the upstream behaviour we work around
    assert verify(parse("(10, -1)"), parse("(3, -1)")) is True
    # our judge rejects it
    assert math_equiv("(10, -1)", "(3, -1)") is False


@pytest.mark.parametrize("pred,gold,want", [
    # pairs (2-tuple)
    ("(3, -1)", "(3, -1)", True),
    ("(10, -1)", "(3, -1)", False),
    ("(18, -18)", "(18,-18)", True),
    ("(18, -18)", "(25, -18)", False),
    # triples
    (r"(\frac{11}{5},\frac{2}{5},5)", r"(\frac{11}{5},\frac{2}{5},5)", True),
    (r"(\frac{18}{5},\frac{2}{5},5)", r"(\frac{11}{5},\frac{2}{5},5)", False),
    (r"(2.2,0.4,5)", r"(\frac{11}{5},\frac{2}{5},5)", True),
    # arity mismatch
    ("(1, 2, 3)", "(1, 2)", False),
    # union-of-intervals: a comma-bearing compound answer that must match on
    # each bound, both raw and restyled
    (r"(-\infty,2)\cup(2,\infty)", r"(-\infty, 2) \cup (2, \infty)", True),
    (r"(-\infty,3)\cup(3,\infty)", r"(-\infty, 2) \cup (2, \infty)", False),
])
def test_tuple_and_union_grading(pred, gold, want):
    assert math_equiv(pred, gold) is want


def test_scalar_unaffected_by_tuple_guard():
    # a bare scalar never enters the tuple path
    assert math_equiv("0.5", r"\frac{1}{2}")
    assert not math_equiv("0.5", r"\frac{1}{3}")


# ---- T11c: empty-gold rows never become false 0-reward samples ------------

def test_row_gold_detects_empty_boxed():
    from meshy.dataset.hendrycks_math import _row_gold
    assert _row_gold({"solution": r"answer is $\boxed{}$."}) is None
    assert _row_gold({"solution": r"so $\boxed{42}$"}) == "42"
    assert _row_gold({"answer": r"\frac12"}) == r"\frac12"
    assert _row_gold({"answer": "  "}) is None


def test_drop_rows_without_gold_filters_by_default_and_raises_strict():
    from datasets import Dataset
    from meshy.dataset.hendrycks_math import (
        GoldAnswerError,
        _drop_rows_without_gold,
    )
    ds = Dataset.from_list([
        {"problem": "a", "solution": r"$\boxed{1}$"},
        {"problem": "b", "solution": r"$\boxed{}$"},   # empty gold
        {"problem": "c", "solution": r"$\boxed{3}$"},
    ])
    kept = _drop_rows_without_gold(ds, strict=False)
    assert len(kept) == 2
    assert [r["problem"] for r in kept] == ["a", "c"]
    with pytest.raises(GoldAnswerError):
        _drop_rows_without_gold(ds, strict=True)


def test_reward_raises_on_empty_gold():
    import types
    from meshy.dataset.hendrycks_math import GoldAnswerError, HendrycksMATH
    s = types.SimpleNamespace(
        messages=[None, {"content": "</think>\boxed{0}"}], ground_truth="")
    with pytest.raises(GoldAnswerError):
        HendrycksMATH.reward(s)
