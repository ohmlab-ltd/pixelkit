"""Boxes and polygons of an augmented copy follow its lens distortion."""
import numpy as np
import pytest
import torch


def test_the_lens_mapping_inverts_the_kernels_sampling(app_module):
    W, H = 320, 240
    cx, cy = (W - 1) / 2, (H - 1) / 2
    for strength in (2, 6, 10):
        k = app_module._lens_k1(strength)
        for ux, uy in [(0.0, 0.0), (0.5, -0.3), (-0.6, 0.6), (0.2, 0.7), (1.0, 1.0), (-1.0, 0.0)]:
            rho2 = ((ux * cx) ** 2 + (uy * cy) ** 2) / (cx * cx + cy * cy)
            f = (1 + k * rho2) / (1 + k)
            # The kernel shows at output point u the source pixel at u * f.
            x, y = (ux * f + 1) * cx, (uy * f + 1) * cy
            ox, oy, shown = app_module._lens_forward_point(x, y, (W, H, k))
            assert shown
            assert ox == pytest.approx((ux + 1) * cx, abs=1e-6)
            assert oy == pytest.approx((uy + 1) * cy, abs=1e-6)


def test_the_barrel_keeps_the_corners_and_pushes_the_edge_middles_out(app_module):
    W, H = 320, 240
    lens = (W, H, app_module._lens_k1(10))
    for corner in [(0.0, 0.0), (W - 1.0, 0.0), (0.0, H - 1.0), (W - 1.0, H - 1.0)]:
        ox, oy, shown = app_module._lens_forward_point(*corner, lens)
        assert shown and (ox, oy) == pytest.approx(corner, abs=1e-6)
    assert app_module._lens_forward_point(W - 1.0, (H - 1) / 2, lens)[2] is False
    assert app_module._lens_forward_point((W - 1) / 2, 0.0, lens)[2] is False


@pytest.mark.parametrize("box", [[230, 30, 270, 70], [20, 170, 70, 225], [140, 100, 180, 140]])
@pytest.mark.parametrize("strength", [1, 3, 6, 10])
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
    # At full strength a box at the middle of an edge is pushed out of the copy.
    full = (W, H, app_module._lens_k1(10))
    assert app_module._apply_matrix_to_box([312.0, 115.0, 319.0, 125.0], None, full) is None
    assert app_module._apply_matrix_to_polys([[[312.0, 115.0], [319.0, 115.0], [319.0, 125.0]]], None, full) == []
    # No matrix and no lens: untouched.
    assert app_module._apply_matrix_to_box([1.0, 2.0, 3.0, 4.0], None) == [1.0, 2.0, 3.0, 4.0]
