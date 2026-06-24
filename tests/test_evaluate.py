"""Tests for src/evaluate.py metric functions, evaluate_checkpoint, and CLI."""
from __future__ import annotations

import numpy as np
import pytest

from src import evaluate


class TestRocAuc:
    def test_perfect_separation_is_one(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        assert evaluate.roc_auc(y, score) == pytest.approx(1.0)

    def test_inverted_separation_is_zero(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.9, 0.8, 0.2, 0.1])
        assert evaluate.roc_auc(y, score) == pytest.approx(0.0)


class TestEqualErrorRate:
    def test_separable_scores_have_zero_eer(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        eer, _ = evaluate.equal_error_rate(y, score)
        assert eer == pytest.approx(0.0, abs=1e-6)

    def test_tied_scores_have_eer_near_half(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        eer, _ = evaluate.equal_error_rate(y, score)
        assert eer == pytest.approx(0.5, abs=0.5)


class TestF1AtThreshold:
    def test_f1_prec_rec_at_known_threshold(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.6, 0.4, 0.9])
        f1, prec, rec = evaluate.f1_at_threshold(y, score, threshold=0.5)
        # preds = [0,1,0,1]; tp=1, fp=1, fn=1
        # prec=0.5, rec=0.5, f1=0.5
        assert prec == pytest.approx(0.5)
        assert rec == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)


class TestConfusion:
    def test_confusion_counts(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        pred = np.array([0, 1, 1, 0, 0, 1])
        cm = evaluate.confusion(y, pred)
        # tn=2, fp=1, fn=1, tp=2
        assert cm == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}


class TestPerProviderRecall:
    def test_per_provider_recall_three_providers(self):
        y = np.array([1, 1, 1, 1, 1, 1])
        pred = np.array([1, 0, 1, 1, 1, 0])
        providers = np.array(["a", "a", "b", "b", "c", "c"])
        result = evaluate.per_provider_recall(y, pred, providers)
        assert result == {"a": pytest.approx(0.5), "b": pytest.approx(1.0), "c": pytest.approx(0.5)}

    def test_skips_provider_with_no_positives(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 1])
        providers = np.array(["a", "a", "b", "b"])
        result = evaluate.per_provider_recall(y, pred, providers)
        assert "a" not in result
        assert result["b"] == pytest.approx(1.0)


class TestMetricBattery:
    def test_battery_assembles_dict(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        providers = np.array(["x", "x", "x", "y", "y", "y"])
        out = evaluate.metric_battery(y, score, providers)
        assert "roc_auc" in out and out["roc_auc"] == pytest.approx(1.0)
        assert "eer" in out and "eer_threshold" in out
        assert "f1" in out and "precision" in out and "recall" in out
        assert "confusion" in out
        assert "per_provider_recall" in out
        assert "n" in out and out["n"] == 6
