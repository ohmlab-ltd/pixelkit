"""An augmented copy's boxes follow its picture: the camera's lens
distortion bends the picture after the perspective and scale/rotation
matrices, and the annotations have to follow it as well."""
import numpy as np
import pytest
import torch


def test_the_lens_mapping_inverts_the_kernels_sampling(app_module):
    W, H = 320, 240
    k1 = app_module._lens_k1(6)
    for ux, uy in [(0.0, 0.0), (0.5, -0.3), (-0.6, 0.6), (0.2, 0.7)]:
        f = 1.0 - k1 * (ux * ux + uy * uy)
        # The kernel shows at output point u the source pixel at u * f.
        x, y = (ux * f + 1) * (W - 1) / 2, (uy * f + 1) * (H - 1) / 2
        ox, oy, shown = app_module._lens_forward_point(x, y, (W, H, k1))
        assert shown
        assert ox == pytest.approx((ux + 1) * (W - 1) / 2, abs=1e-6)
        assert oy == pytest.approx((uy + 1) * (H - 1) / 2, abs=1e-6)
    # At full strength the warp folds: a corner pixel shows nowhere.
    assert app_module._lens_forward_point(0.0, 0.0, (W, H, app_module._lens_k1(10)))[2] is False


@pytest.mark.parametrize("box", [[230, 30, 270, 70], [20, 170, 70, 225], [140, 100, 180, 140]])
@pytest.mark.parametrize("strength", [1, 2, 3])
def test_a_box_follows_its_object_through_the_lens_warp(app_module, box, strength):
    W, H = 320, 240
    img = torch.zeros(3, H, W)
    img[:, box[1]:box[3], box[0]:box[2]] = 1.0
    out = app_module._aug_apply(img, 0, 0, 0, 0, 0, 1, lens_distortion=strength)
    ys, xs = np.nonzero(out[0].numpy() > 0.5)
    seen = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    lens = (W, H, app_module._lens_k1(strength))
    warped = app_module._clip_box_to(app_module._apply_matrix_to_box([float(v) for v in box], None, lens), W, H)
    assert warped == pytest.approx(seen, abs=1.5)


def test_polygons_follow_and_what_the_lens_hides_goes(app_module):
    W, H = 320, 240
    lens = (W, H, app_module._lens_k1(2))
    (poly,) = app_module._apply_matrix_to_polys([[[230.0, 30.0], [270.0, 30.0], [270.0, 70.0]]], None, lens)
    assert poly[0] == pytest.approx(app_module._lens_forward_point(230.0, 30.0, lens)[:2], abs=1e-9)
    # At full strength a box wholly past the fold is not in the copy.
    full = (W, H, app_module._lens_k1(10))
    assert app_module._apply_matrix_to_box([0.0, 0.0, 12.0, 12.0], None, full) is None
    assert app_module._apply_matrix_to_polys([[[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]]], None, full) == []
    # No matrix and no lens: untouched.
    assert app_module._apply_matrix_to_box([1.0, 2.0, 3.0, 4.0], None) == [1.0, 2.0, 3.0, 4.0]
