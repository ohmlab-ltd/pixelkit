"""The label a detection gets from the reference embeddings of a dataset
with reference images."""
import numpy as np


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _refs():
    rng = np.random.default_rng(0)
    base = {"bolt": _unit([1, 0, 0, 0]), "nut": _unit([0, 1, 0, 0])}
    return {k: np.stack([_unit(b + rng.normal(0, 0.05, 4)) for _ in range(4)]) for k, b in base.items()}


def test_the_nearest_references_choose_the_label(app_module):
    v = app_module._v2_resolve_label_specific(
        _unit([0.95, 0.1, 0, 0]).tolist(), None, _refs(), {"bolt": "Bolt", "nut": "Nut"})
    assert v["pred_label"] == "Bolt" and v["pred_source"] == "embed"
    assert not v["rejected"] and not v["ambiguous"]
    assert v["sims"]["Bolt"] > v["sims"]["Nut"]


def test_a_detection_like_no_reference_is_rejected(app_module):
    v = app_module._v2_resolve_label_specific(
        _unit([0, 0, 1, 0.2]).tolist(), None, _refs(), {"bolt": "bolt", "nut": "nut"})
    assert v["rejected"] and v["reject_reason"] == "embed"


def test_a_detection_between_two_labels_is_ambiguous(app_module):
    v = app_module._v2_resolve_label_specific(
        _unit([1, 1, 0, 0]).tolist(), None, _refs(), {"bolt": "bolt", "nut": "nut"},
        ambiguous_margin=0.08)
    assert v["ambiguous"] and not v["rejected"]


def test_without_references_the_segmenters_label_stands(app_module):
    v = app_module._v2_resolve_label_specific([], "washer", {}, {})
    assert v["pred_label"] == "washer" and v["pred_source"] == "gd" and not v["rejected"]


def test_a_drawn_box_is_scored_against_the_references(app_module):
    by_label = {k: list(arr) for k, arr in _refs().items()}
    v, _t = app_module._charlie_resolve_from_refs(
        _unit([0.1, 0.95, 0, 0]).tolist(), None, by_label, ["bolt", "nut"], refs_patch={})
    assert v["pred_label"] == "nut" and not v["rejected"]
