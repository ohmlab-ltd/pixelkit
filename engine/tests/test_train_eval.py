"""Threshold calibration, class report, confusion matrix and failure gallery, without a model."""
import numpy as np
import pytest


def _preds_gts():
    # Image 0: one dog GT; a confident correct dog and a weak spurious cat.
    # Image 1: one cat GT called a dog at 0.6, and a stray 0.3 dog box.
    # Image 2: a dog GT the model misses.
    dog, cat = 0, 1
    preds = [
        np.array([[10, 10, 50, 50, 0.90, dog], [70, 70, 90, 90, 0.30, cat]], np.float32),
        np.array([[10, 10, 50, 50, 0.60, dog], [60, 5, 80, 20, 0.30, dog]], np.float32),
        np.zeros((0, 6), np.float32),
    ]
    gts = [
        np.array([[12, 12, 50, 50, dog]], np.float32),
        np.array([[10, 10, 52, 52, cat]], np.float32),
        np.array([[30, 30, 60, 60, dog]], np.float32),
    ]
    return preds, gts


def test_match_kinds():
    from train_eval import match_image

    preds, gts = _preds_gts()
    kinds = sorted(k for k, _p, _g in match_image(preds[0], gts[0], conf=0.25))
    assert kinds == ["fp", "tp"]
    kinds = sorted(k for k, _p, _g in match_image(preds[1], gts[1], conf=0.25))
    assert kinds == ["fp", "wrong"]
    kinds = [k for k, _p, _g in match_image(preds[2], gts[2], conf=0.25)]
    assert kinds == ["fn"]
    # Raising the threshold drops the weak boxes.
    kinds = sorted(k for k, _p, _g in match_image(preds[0], gts[0], conf=0.5))
    assert kinds == ["tp"]


def test_same_class_box_is_preferred_over_a_better_overlap_of_another_class():
    from train_eval import match_image

    dog, cat = 0, 1
    pred = np.array([[10, 10, 50, 50, 0.9, dog]], np.float32)
    gt = np.array([[10, 10, 50, 50, cat],       # perfect overlap, wrong class
                   [12, 12, 50, 50, dog]], np.float32)  # slightly worse, right class
    m = match_image(pred, gt, conf=0.1)
    assert ("tp", 0, 1) in m
    assert sorted(k for k, _p, _g in m) == ["fn", "tp"]


def test_calibrated_threshold_separates_good_from_weak_boxes():
    from train_eval import calibrate_threshold

    preds, gts = _preds_gts()
    cal = calibrate_threshold(preds, gts, n_classes=2)
    assert cal["calibrated"] is True
    # F1 peaks at 0.5 on the plateau from 0.6 to 0.9; the calibrator picks its middle.
    assert cal["conf"] == pytest.approx(0.75, abs=0.011)
    assert cal["precision"] == 1.0
    assert cal["recall"] == pytest.approx(1 / 3, abs=1e-3)
    assert len(cal["sweep"]) > 10 and cal["sweep"][0]["conf"] == 0.05


def test_calibration_falls_back_when_the_model_finds_nothing():
    from train_eval import calibrate_threshold

    preds = [np.zeros((0, 6), np.float32)]
    gts = [np.array([[0, 0, 10, 10, 0]], np.float32)]
    cal = calibrate_threshold(preds, gts, n_classes=1)
    assert cal["calibrated"] is False and cal["conf"] == 0.25


def test_class_report_and_confusion_matrix():
    from train_eval import class_report, confusion_matrix

    preds, gts = _preds_gts()
    rep = class_report(preds, gts, conf=0.25, classes=["dog", "cat"])
    dog, cat = rep
    # dog: 1 tp (img0), 2 fp (wrong-class box in img1 counts against dog),
    # 1 fn (img2). cat: 0 tp, 1 fp (img0 weak), 1 fn (img1 called dog).
    assert (dog["tp"], dog["fp"], dog["fn"], dog["support"]) == (1, 2, 1, 2)
    assert (cat["tp"], cat["fp"], cat["fn"], cat["support"]) == (0, 1, 1, 1)
    assert dog["precision"] == pytest.approx(1 / 3, abs=1e-3) and dog["recall"] == 0.5
    assert cat["precision"] == 0.0 and cat["recall"] == 0.0

    cm = confusion_matrix(preds, gts, conf=0.25, n_classes=2)
    # rows truth [dog, cat, bg], cols predicted [dog, cat, bg]
    assert cm[0][0] == 1          # dog found as dog
    assert cm[1][0] == 1          # cat called dog
    assert cm[0][2] == 1          # dog missed
    assert cm[2][0] == 1 and cm[2][1] == 1  # spurious dog and cat
    assert sum(map(sum, cm)) == 5


def test_failures_are_ranked_worst_first_and_rendered(tmp_path):
    from train_eval import rank_failures, write_failure_gallery

    preds, gts = _preds_gts()
    rows = rank_failures(preds, gts, conf=0.25)
    # image 1 (wrong + fp) outranks image 2 (one miss) outranks image 0 (one fp)
    assert [r["index"] for r in rows] == [1, 2, 0]

    def load(i):
        return np.full((100, 100, 3), 120, np.uint8)

    gallery = write_failure_gallery(preds, gts, 0.25, ["dog", "cat"], ["a.jpg", "b.jpg", "c.jpg"],
                                    load, tmp_path / "failures", limit=2)
    assert [g["image"] for g in gallery] == ["b.jpg", "c.jpg"]
    assert (tmp_path / "failures" / gallery[0]["file"]).stat().st_size > 500
    assert gallery[0]["width"] == 384 and gallery[0]["height"] == 384
    kinds = sorted(it["kind"] for it in gallery[0]["items"])
    assert kinds == ["fp", "wrong"]
    assert gallery[0]["items"][0]["box"][2] <= 100  # boxes reported in model space


def test_a_stronger_wrong_class_box_does_not_steal_the_truth_from_the_right_one():
    from train_eval import class_report, match_image

    dog, cat = 0, 1
    pred = np.array([[10, 10, 50, 50, 0.9, cat],    # confident, wrong class
                     [10, 10, 50, 50, 0.8, dog]], np.float32)  # right class, weaker
    gt = np.array([[10, 10, 50, 50, dog]], np.float32)
    m = match_image(pred, gt, conf=0.1)
    assert ("tp", 1, 0) in m and ("fp", 0, -1) in m
    rep = class_report([pred], [gt], 0.1, ["dog", "cat"])
    assert rep[0]["tp"] == 1 and rep[0]["fn"] == 0 and rep[0]["recall"] == 1.0
    assert rep[1]["fp"] == 1
