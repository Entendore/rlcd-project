import numpy as np
import pytest

from rlcd.core.primitives import (PrimitiveType, Question,
                                  choice_confidence_from_distribution,
                                  noul_confidence_from_probability)
from rlcd.evaluation.metrics import full_report


def test_choice_requires_criteria():
    with pytest.raises(ValueError):
        Question(id="x", type=PrimitiveType.CHOICE, instructions="?")


def test_score_requires_scale():
    with pytest.raises(ValueError):
        Question(id="y", type=PrimitiveType.SCORE, instructions="?")


def test_confidence_functions():
    assert choice_confidence_from_distribution({"a": 0.9, "b": 0.1}) == pytest.approx(0.8)
    assert noul_confidence_from_probability(0.99) == pytest.approx(0.98)
    assert noul_confidence_from_probability(0.5) == pytest.approx(0.0)


def test_validate_answer_space():
    q = Question(id="d", type=PrimitiveType.CHOICE, instructions="?",
                 criteria={"a": "x", "b": "y"})
    assert q.validate_answer_space("a") and not q.validate_answer_space("c")


def test_perfect_calibration_zero_ece():
    from rlcd.evaluation.metrics import expected_calibration_error
    confs = np.array([0.9] * 100)
    corrects = np.array([1] * 90 + [0] * 10)
    assert expected_calibration_error(confs, corrects) < 0.01


def test_report_keys():
    rng = np.random.default_rng(1)
    c = rng.random(500)
    y = (rng.random(500) < c).astype(float)
    assert {"ece", "mce", "brier", "nll", "accuracy"} <= set(full_report(c, y))
