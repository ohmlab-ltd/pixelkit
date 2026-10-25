"""What augmented copies were made from, and when they are out of date."""
import aug_source


def _imp(**kw):
    return {"detections": [{"box": [1, 2, 30, 40], "label": "Bolt"}], **kw}


def test_edits_win_over_detections_and_an_explicit_empty_edit_means_none():
    assert aug_source.source_boxes(_imp()) == [{"label": "bolt", "box": [1.0, 2.0, 30.0, 40.0], "mask": ""}]
    edited = _imp(editedBoxes=[{"x0": 5, "y0": 5, "x1": 9, "y1": 9, "label": "nut"}])
    assert [b["label"] for b in aug_source.source_boxes(edited)] == ["nut"]
    assert aug_source.source_boxes(_imp(editedBoxes=[], editedBoxesSet=True)) == []
    assert len(aug_source.source_boxes(_imp(editedBoxes=[]))) == 1


def test_any_change_to_a_box_or_its_mask_makes_copies_out_of_date():
    imp = _imp()
    anno = {"source": aug_source.source_boxes(imp)}
    assert aug_source.is_current(anno, imp)
    assert aug_source.is_current({"copies": {}}, imp), "copies from before the record count as current"
    assert not aug_source.is_current(anno, _imp(detections=[{"box": [1, 2, 31, 40], "label": "bolt"}]))
    masked = _imp(detections=[{"box": [1, 2, 30, 40], "label": "bolt",
                               "mask": {"polygons": [[[1, 2], [30, 2], [30, 40]]]}}])
    assert not aug_source.is_current(anno, masked)
    two = _imp(detections=[{"box": [5, 5, 9, 9], "label": "nut"}, {"box": [1, 2, 30, 40], "label": "bolt"}])
    assert aug_source.is_current({"source": aug_source.source_boxes(two)},
                                 {"detections": list(reversed(two["detections"]))}), "order does not matter"


def test_a_rename_or_a_purge_keeps_the_record_in_step():
    anno = {"source": aug_source.source_boxes(_imp(detections=[
        {"box": [1, 2, 30, 40], "label": "bolt"}, {"box": [5, 5, 9, 9], "label": "nut"}]))}
    assert aug_source.relabel_source(anno, lambda lab: "Screw" if lab == "bolt" else lab)
    assert sorted(b["label"] for b in anno["source"]) == ["nut", "screw"]
    assert aug_source.relabel_source(anno, lambda lab: None if lab == "nut" else lab)
    assert [b["label"] for b in anno["source"]] == ["screw"]
    assert not aug_source.relabel_source(anno, lambda lab: lab)
    assert not aug_source.relabel_source({"copies": {}}, lambda lab: None)


def test_rejected_detections_are_not_copied():
    imp = {"detections": [{"box": [1, 2, 30, 40], "label": "bolt"},
                          {"box": [5, 5, 9, 9], "label": "nut", "rejected": True}]}
    assert [b["label"] for b in aug_source.source_boxes(imp)] == ["bolt"]
