import os
import unittest
import io as io_std
from contextlib import redirect_stdout
import numpy as np

import linalg
import homography
import warp
import blend

class TestInterfaceContract(unittest.TestCase):
    def test_linalg_stubs_importable(self):
        """Contract: rref, solve, inverse, det, rank, nullspace_vector in linalg.py"""
        for fn_name in ['rref', 'solve', 'inverse', 'det', 'rank', 'nullspace_vector']:
            self.assertTrue(hasattr(linalg, fn_name), f"Missing {fn_name} in linalg.py")
            self.assertTrue(callable(getattr(linalg, fn_name)), f"{fn_name} is not callable")

        # Basic functionality smoke tests
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 5.0])
        x = linalg.solve(A, b)
        self.assertTrue(np.allclose(A @ x, b))

        invA = linalg.inverse(A)
        self.assertTrue(np.allclose(invA @ A, np.eye(2)))

        self.assertAlmostEqual(linalg.det(A), 5.0)
        self.assertEqual(linalg.rank(A), 2)

        null_v = linalg.nullspace_vector(np.array([[1.0, 2.0], [2.0, 4.0]]))
        self.assertAlmostEqual(np.linalg.norm(np.array([[1.0, 2.0], [2.0, 4.0]]) @ null_v), 0.0, places=5)

    def test_homography_stubs_importable(self):
        """Contract: normalize_points, solve_homography, solve_homography_lsq,
        algebraic_residual, transfer_error in homography.py"""
        fns = [
            'normalize_points',
            'solve_homography',
            'solve_homography_lsq',
            'algebraic_residual',
            'transfer_error'
        ]
        for fn_name in fns:
            self.assertTrue(hasattr(homography, fn_name), f"Missing {fn_name} in homography.py")
            self.assertTrue(callable(getattr(homography, fn_name)), f"{fn_name} is not callable")

    def test_warp_stubs_importable(self):
        """Contract: check_h, canvas_bounds, warp_image, edge_weights,
        feather_blend, ransac_homography in warp.py"""
        fns = [
            'check_h',
            'canvas_bounds',
            'warp_image',
            'edge_weights',
            'feather_blend',
            'ransac_homography'
        ]
        for fn_name in fns:
            self.assertTrue(hasattr(warp, fn_name), f"Missing {fn_name} in warp.py")
            self.assertTrue(callable(getattr(warp, fn_name)), f"{fn_name} is not callable")


class TestNpLinalgBan(unittest.TestCase):
    def test_np_linalg_banned_outside_allowed_files(self):
        """Acceptance: A test fails if np.linalg appears outside linalg.py and tests.py."""
        allowed_files = {'linalg.py', 'tests.py'}
        repo_root = os.path.dirname(os.path.abspath(__file__))
        
        violations = []
        for root, dirs, files in os.walk(repo_root):
            # Ignore virtualenv, git, cache, and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__')]
            for file in files:
                if file.endswith('.py') and file not in allowed_files:
                    path = os.path.join(root, file)
                    rel_path = os.path.relpath(path, repo_root)
                    with open(path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, start=1):
                            if 'np.linalg' in line or 'numpy.linalg' in line:
                                violations.append(f"{rel_path}:{line_num}: {line.strip()}")

        self.assertEqual(violations, [], f"Found disallowed usages of np.linalg outside linalg.py and tests.py:\n" + "\n".join(violations))


class TestEndToEndKnownH(unittest.TestCase):
    def test_known_h_synthetic(self):
        """Acceptance: One known-H check passes end to end on a synthetic image."""
        # 100x100 grayscale synthetic image with a square
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:80, 20:80] = 180

        src = np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]])
        dst = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])

        # Ground truth homography mapping [20, 80] to [0, 100]:
        # x' = (100 / 60) * (x - 20) = (5/3)*x - 100/3
        # y' = (100 / 60) * (y - 20) = (5/3)*y - 100/3
        scale = 100.0 / 60.0
        shift = -20.0 * scale
        expected_H = np.array([
            [scale, 0.0, shift],
            [0.0, scale, shift],
            [0.0, 0.0, 1.0]
        ])

        H = homography.solve_homography(src, dst)
        self.assertTrue(np.allclose(H, expected_H, atol=1e-5), f"H mismatch:\n{H}\nExpected:\n{expected_H}")

        # Check warp_image output shape and contract return values: (out, valid, x, y)
        out, valid, x, y = warp.warp_image(img, H, (100, 100), (0, 0))
        self.assertEqual(out.shape, (100, 100))
        self.assertEqual(valid.shape, (100, 100))
        self.assertEqual(x.shape, (100, 100))
        self.assertEqual(y.shape, (100, 100))

        # The center of the warped image must contain the source square's pixel value
        self.assertEqual(out[50, 50], 180)
        self.assertTrue(valid[50, 50])


class TestV08Points(unittest.TestCase):
    def test_json_round_trip_exact(self):
        import tempfile
        import points

        named = {
            "src": np.array([[1.0, 2.5], [10.0, 0.0], [3.0, 4.0]], dtype=float),
            "dst": np.array([[5.0, 6.0], [7.5, 8.0], [9.0, 1.0]], dtype=float),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pts.json")
            points.save_points(path, named)
            loaded = points.load_points(path)
        self.assertEqual(set(loaded.keys()), set(named.keys()))
        for key in named:
            self.assertTrue(np.array_equal(loaded[key], named[key]), key)

    def test_calibration_markers_at_known_pixels(self):
        import points

        img, pts = points.make_calibration_image(height=60, width=40)
        self.assertEqual(img.shape[0], 60)
        self.assertEqual(img.shape[1], 40)
        for x, y in pts:
            pixel = img[int(y), int(x)]
            self.assertFalse(np.array_equal(pixel, [20, 20, 20]))

    def test_mark_points_stamps_the_right_pixel(self):
        import points

        img = np.zeros((60, 40, 3), dtype=np.uint8)
        pts = np.array([[7.0, 11.0]], dtype=float)
        stamped = points.mark_points(img, pts)
        self.assertTrue(np.array_equal(stamped[11, 7], points.STAMP_RGB))
        self.assertTrue(np.array_equal(stamped[11, 6], [0, 0, 0]))
        self.assertTrue(np.array_equal(stamped[10, 7], [0, 0, 0]))
        self.assertTrue(np.array_equal(img[11, 7], [0, 0, 0]))


class TestV09Imgops(unittest.TestCase):
    def _asymmetric_rgb(self):
        img = np.zeros((60, 40, 3), dtype=np.uint8)
        img[0, 0] = [255, 0, 0]
        img[0, -1] = [0, 255, 0]
        img[-1, 0] = [0, 0, 255]
        img[-1, -1] = [255, 255, 0]
        img[20, 10] = [0, 255, 255]
        return img

    def test_flip_twice_is_identity(self):
        import imgops

        img = self._asymmetric_rgb()
        self.assertTrue(np.array_equal(imgops.flip_h(imgops.flip_h(img)), img))
        self.assertTrue(np.array_equal(imgops.flip_v(imgops.flip_v(img)), img))

    def test_rotate90_four_times_is_identity(self):
        import imgops

        img = self._asymmetric_rgb()
        out = img
        for _ in range(4):
            out = imgops.rotate90(out, k=1)
        self.assertTrue(np.array_equal(out, img))

    def test_rotate90_ccw_matches_numpy_on_marked_pixel(self):
        import imgops

        img = np.zeros((60, 40), dtype=np.uint8)
        img[11, 7] = 200
        got = imgops.rotate90(img, k=1)
        expected = np.rot90(img, k=1)
        self.assertEqual(got.shape, (40, 60))
        self.assertTrue(np.array_equal(got, expected))
        loc = np.argwhere(got == 200)
        self.assertEqual(len(loc), 1)
        self.assertEqual(tuple(loc[0]), tuple(np.argwhere(expected == 200)[0]))

    def test_resize_rows_sum_to_one(self):
        import imgops

        r, c = imgops.resize_matrices(60, 40, 15, 10)
        self.assertTrue(np.allclose(r.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(c.sum(axis=1), 1.0))

    def test_constant_image_stays_constant_after_resize(self):
        import imgops

        for value in (17, 200):
            img = np.full((60, 40), value, dtype=np.uint8)
            out = imgops.resize(img, 24, 18)
            self.assertEqual(out.shape, (24, 18))
            self.assertTrue(np.all(out == value))

    def test_grayscale_pure_colors_near_weight_times_255(self):
        import imgops

        red = np.full((4, 5, 3), (255, 0, 0), dtype=np.uint8)
        green = np.full((4, 5, 3), (0, 255, 0), dtype=np.uint8)
        blue = np.full((4, 5, 3), (0, 0, 255), dtype=np.uint8)
        white = np.full((4, 5, 3), (255, 255, 255), dtype=np.uint8)
        self.assertTrue(np.all(np.abs(imgops.grayscale(red).astype(float) - imgops.WEIGHTS[0] * 255) <= 0.5))
        self.assertTrue(np.all(np.abs(imgops.grayscale(green).astype(float) - imgops.WEIGHTS[1] * 255) <= 0.5))
        self.assertTrue(np.all(np.abs(imgops.grayscale(blue).astype(float) - imgops.WEIGHTS[2] * 255) <= 0.5))
        self.assertTrue(np.all(imgops.grayscale(white) == 255))

    def test_exif_tags_1_to_8_come_out_upright(self):
        import imgops

        upright = self._asymmetric_rgb()
        top_left = upright[0, 0].copy()

        def invert_orientation(img, tag):
            if tag == 1:
                return img.copy()
            if tag == 2:
                return imgops.flip_h(img)
            if tag == 3:
                return imgops.rotate90(img, k=2)
            if tag == 4:
                return imgops.flip_v(img)
            if tag == 5:
                return imgops.transpose_img(img)
            if tag == 6:
                return imgops.rotate90(img, k=1)
            if tag == 7:
                return imgops.rotate90(imgops.transpose_img(img), k=2)
            if tag == 8:
                return imgops.rotate90(img, k=-1)
            raise AssertionError(tag)

        for tag in range(1, 9):
            stored = invert_orientation(upright, tag)
            recovered = imgops.apply_exif_orientation(stored, tag)
            self.assertEqual(recovered.shape, upright.shape, "tag {}".format(tag))
            self.assertTrue(np.array_equal(recovered, upright), "tag {}".format(tag))
            self.assertTrue(np.array_equal(recovered[0, 0], top_left), "tag {}".format(tag))

    def test_exif_table_matches_pillow_exif_transpose(self):
        """Independent check: our table must agree with Pillow's own EXIF handling."""
        from PIL import Image, ImageOps
        import imgops

        stored = self._asymmetric_rgb()
        for tag in range(1, 9):
            pil_im = Image.fromarray(stored)
            pil_im.getexif()[274] = tag
            expected = np.array(ImageOps.exif_transpose(pil_im))
            got = imgops.apply_exif_orientation(stored, tag)
            self.assertEqual(got.shape, expected.shape, "tag {}".format(tag))
            self.assertTrue(np.array_equal(got, expected), "tag {}".format(tag))


class TestPointsChecksAndRectify(unittest.TestCase):
    def test_collinearity_and_corner_order(self):
        import points

        self.assertTrue(points.are_collinear((0, 0), (1, 1), (2, 2)))
        self.assertFalse(points.are_collinear((0, 0), (1, 0), (0, 1)))
        clockwise = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]])
        swapped = np.array([[0.0, 0.0], [10.0, 8.0], [10.0, 0.0], [0.0, 8.0]])
        degenerate = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [0.0, 8.0]])
        self.assertTrue(points.corner_order_ok(clockwise))
        self.assertFalse(points.corner_order_ok(swapped))
        self.assertFalse(points.corner_order_ok(degenerate))

    def test_corner_order_winding_option(self):
        import points

        clockwise = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]])
        counter = clockwise[[0, 3, 2, 1]]
        self.assertTrue(points.corner_order_ok(clockwise, winding="clockwise"))
        self.assertFalse(points.corner_order_ok(clockwise, winding="counterclockwise"))
        self.assertTrue(points.corner_order_ok(counter, winding="counterclockwise"))
        self.assertFalse(points.corner_order_ok(counter, winding="clockwise"))
        self.assertTrue(points.corner_order_ok(counter))  # default accepts either winding

    def test_rectify_rejects_counter_clockwise_corners(self):
        import tempfile
        import demo
        import io_utils
        import points

        img = np.zeros((60, 40, 3), dtype=np.uint8)
        counter = np.array([[8.0, 10.0], [8.0, 50.0], [32.0, 50.0], [32.0, 10.0]], dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "quad.png")
            json_path = os.path.join(tmp, "corners.json")
            io_utils.save_png(image_path, img)
            points.save_points(json_path, {"corners": counter})
            with self.assertRaises(ValueError):
                demo.rectify(
                    image_path,
                    json_path,
                    out_before=os.path.join(tmp, "before.png"),
                    out_after=os.path.join(tmp, "after.png"),
                )

    def test_rectify_smoke_prints_h33_one(self):
        import tempfile
        import demo
        import io_utils
        import points

        img = np.zeros((60, 40, 3), dtype=np.uint8)
        img[10:50, 8:32] = (200, 40, 40)
        corners = np.array(
            [[8.0, 10.0], [32.0, 10.0], [32.0, 50.0], [8.0, 50.0]],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "quad.png")
            json_path = os.path.join(tmp, "corners.json")
            before = os.path.join(tmp, "before.png")
            after = os.path.join(tmp, "after.png")
            io_utils.save_png(image_path, img)
            points.save_points(json_path, {"corners": corners})
            with redirect_stdout(io_std.StringIO()):
                h_mat, out = demo.rectify(image_path, json_path, out_before=before, out_after=after)
            self.assertTrue(os.path.isfile(before))
            self.assertTrue(os.path.isfile(after))
        self.assertTrue(np.isfinite(h_mat).all())
        self.assertAlmostEqual(h_mat[2, 2], 1.0, places=6)
        self.assertEqual(out.ndim, 3)

    def test_open_image_missing_exif_is_orientation_1(self):
        import tempfile
        import io_utils

        img = np.zeros((8, 6, 3), dtype=np.uint8)
        img[0, 0] = (9, 8, 7)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plain.png")
            io_utils.save_png(path, img)
            loaded = io_utils.open_image(path)
            self.assertEqual(io_utils.read_exif_orientation(path), 1)
        self.assertTrue(np.array_equal(loaded[0, 0], [9, 8, 7]))
        loaded[0, 0] = (1, 2, 3)
class TestWarpV07(unittest.TestCase):
    """V0.7: warp exactness, corners, canvas cap, no NaN, check_h guard."""

    def test_identity_h_exact(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(40, 50), dtype=np.uint8)
        H = np.eye(3)
        canvas, offset = warp.canvas_bounds([H], [img.shape])
        self.assertEqual(canvas, (40, 50))
        np.testing.assert_allclose(offset, [0.0, 0.0])

        out, valid, sx, sy = warp.warp_image(img, H, canvas, offset)
        np.testing.assert_array_equal(out, img)
        self.assertTrue(valid.all())
        gx, gy = np.meshgrid(np.arange(50), np.arange(40))
        np.testing.assert_allclose(sx, gx)
        np.testing.assert_allclose(sy, gy)

    def test_integer_translation_exact(self):
        img = np.arange(100 * 80, dtype=np.uint8).reshape(100, 80)
        tx, ty = 7, -13
        H = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        canvas, offset = warp.canvas_bounds([H], [img.shape])
        self.assertEqual(canvas, (100, 80))
        np.testing.assert_allclose(offset, [tx, ty])

        out, valid, _, _ = warp.warp_image(img, H, canvas, offset)
        np.testing.assert_array_equal(out, img)
        self.assertTrue(valid.all())

    def test_canvas_bounds_matches_mapped_corners(self):
        # Scale by 2 about origin plus translate: dst = 2*src + (10, -5)
        H = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0], [0.0, 0.0, 1.0]])
        shape = (30, 40)  # h=30, w=40
        canvas, offset = warp.canvas_bounds([H], [shape])

        corners = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=float)
        mapped = corners @ H[:2, :2].T + H[:2, 2]
        np.testing.assert_allclose(offset, mapped.min(axis=0))
        self.assertEqual(canvas, (int(np.ceil(mapped[:, 1].max() - mapped[:, 1].min())),
                                  int(np.ceil(mapped[:, 0].max() - mapped[:, 0].min()))))

    def test_canvas_rejects_oversize_bounds(self):
        # Reviewer regression: a valid H can exceed the canvas cap; bounds
        # must be rejected rather than silently cropped.
        H = np.array([[1.0e6, 0.0, 0.0], [0.0, 1.0e6, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaises(homography.HomographyError):
            warp.canvas_bounds([H], [(100, 100)])
        # The reviewer's example: 2x100 image scaled by 250 in x.
        H2 = np.array([[250.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaises(homography.HomographyError):
            warp.canvas_bounds([H2], [(2, 100)])

    def test_no_nan_or_inf(self):
        # H whose horizon crosses the canvas: w = 0 inside the image.
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.02, 0.0, 1.0]])
        img = np.ones((100, 100), dtype=np.uint8)
        # w' = 0.02*(x-100) + 1 vanishes at canvas x = 50, inside the canvas.
        out, valid, sx, sy = warp.warp_image(img, H, (200, 200), (-100.0, -100.0))
        for arr in (out, sx, sy):
            self.assertTrue(np.all(np.isfinite(arr)))
        # Some pixels must be invalid (the horizon region)
        self.assertFalse(valid.all())

    def test_check_h_raises_on_mixed_sign_w(self):
        # w = -0.02*x + 1: +1 at x=0, -1 at x=100 -> sign change across image.
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.02, 0.0, 1.0]])
        with self.assertRaises(homography.HomographyError):
            warp.check_h(H, (100, 100))
        # Same-sign w passes.
        warp.check_h(np.eye(3), (100, 100))
        warp.check_h(np.array([[2.0, 0, 5], [0, 2.0, 5], [0, 0, 1.0]]), (100, 100))

    def test_canvas_bounds_propagates_check_h(self):
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.02, 0.0, 1.0]])
        with self.assertRaises(homography.HomographyError):
            warp.canvas_bounds([H], [(100, 100)])

    def test_check_h_rejects_singular(self):
        # w = 1 at every corner, but H is singular: inverse warp impossible.
        H = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaises(homography.HomographyError):
            warp.check_h(H, (100, 100))
        with self.assertRaises(homography.HomographyError):
            warp.canvas_bounds([H], [(100, 100)])
        with self.assertRaises(homography.HomographyError):
            warp.warp_image(np.zeros((10, 10), dtype=np.uint8), H, (10, 10), (0, 0))

    def test_check_h_accepts_small_scale(self):
        # Reviewer regression: det = 1e-14 but H is invertible; a fixed
        # determinant threshold would wrongly reject this downscale.
        H = np.diag([1e-7, 1e-7, 1.0])
        warp.check_h(H, (100, 100))
        canvas, offset = warp.canvas_bounds([H], [(100, 100)])
        self.assertEqual(canvas, (1, 1))

    def test_check_h_rejects_boundary_horizon(self):
        # w = -0.01*x + 1 is exactly 0 at x = 100 (the right image edge).
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.01, 0.0, 1.0]])
        with self.assertRaises(homography.HomographyError):
            warp.check_h(H, (100, 100))
        with self.assertRaises(homography.HomographyError):
            warp.canvas_bounds([H], [(100, 100)])


class TestHomographyExact(unittest.TestCase):
    """V0.4, V0.5, V0.6 tests for exact 4-point homography solver."""

    def test_v04_random_mild_perspective_100_trials(self):
        """V0.4: 100 trials with seeded RNG.
        Mild-perspective H with perturbations and convex quad.
        Assert np.allclose(rtol=1e-8, atol=1e-10) on H and transfer error < 1e-8 px.
        """
        rng = np.random.default_rng(20261110)
        worst_te = 0.0
        worst_h_diff = 0.0

        for trial in range(100):
            H = np.eye(3)
            H[0, 0] += rng.uniform(-0.15, 0.15)
            H[1, 1] += rng.uniform(-0.15, 0.15)
            H[0, 1] += rng.uniform(-0.15, 0.15)
            H[1, 0] += rng.uniform(-0.15, 0.15)
            H[0, 2] += rng.uniform(-40, 40)
            H[1, 2] += rng.uniform(-40, 40)
            H[2, 0] += rng.uniform(-8e-4, 8e-4)
            H[2, 1] += rng.uniform(-8e-4, 8e-4)
            H[2, 2] = 1.0

            src = np.array([
                [100.0 + rng.uniform(-20, 20), 100.0 + rng.uniform(-20, 20)],
                [600.0 + rng.uniform(-20, 20), 100.0 + rng.uniform(-20, 20)],
                [600.0 + rng.uniform(-20, 20), 400.0 + rng.uniform(-20, 20)],
                [100.0 + rng.uniform(-20, 20), 400.0 + rng.uniform(-20, 20)]
            ])

            src_h = np.hstack([src, np.ones((4, 1))])
            dst_h = (H @ src_h.T).T
            dst = dst_h[:, :2] / dst_h[:, 2:3]

            H_est = homography.solve_homography(src, dst)
            self.assertTrue(np.allclose(H, H_est, rtol=1e-8, atol=1e-10))

            te = homography.transfer_error(H_est, src, dst)
            max_te = np.max(te)
            worst_te = max(worst_te, max_te)
            worst_h_diff = max(worst_h_diff, np.max(np.abs(H - H_est)))
            self.assertTrue(np.all(te < 1e-8))

        print(f"\nV0.4 100 trials worst case: max transfer error = {worst_te:.3e} px, max |H - H_est| = {worst_h_diff:.3e}")

    def test_v05_identity_and_affine_rhombus(self):
        """V0.5: Identical large coordinates give H within 1e-9 of identity.
        A rectangle sheared into a parallelogram gives |h31| and |h32| under 1e-9.
        """
        # Identical large coordinates
        src = np.array([
            [1200.0, 3400.0],
            [5600.0, 3500.0],
            [5500.0, 7800.0],
            [1100.0, 7900.0]
        ])
        dst = src.copy()
        H_id = homography.solve_homography(src, dst)
        self.assertTrue(np.all(np.abs(H_id - np.eye(3)) < 1e-9))

        # Sheared parallelogram (rhombus)
        src_rect = np.array([
            [100.0, 100.0],
            [500.0, 100.0],
            [500.0, 300.0],
            [100.0, 300.0]
        ])
        A_aff = np.array([
            [1.0, 0.4, 35.0],
            [0.0, 1.0, -15.0],
            [0.0, 0.0, 1.0]
        ])
        src_h = np.hstack([src_rect, np.ones((4, 1))])
        dst_aff = (A_aff @ src_h.T).T[:, :2]
        H_aff = homography.solve_homography(src_rect, dst_aff)

        self.assertLess(abs(H_aff[2, 0]), 1e-9)
        self.assertLess(abs(H_aff[2, 1]), 1e-9)
        self.assertAlmostEqual(H_aff[2, 2], 1.0, places=9)

    def test_v06_collinear_swapped_and_horizon_failures(self):
        """V0.6: Failure cases raise HomographyError with distinct messages."""
        valid_quad = np.array([
            [100.0, 100.0],
            [500.0, 100.0],
            [500.0, 400.0],
            [100.0, 400.0]
        ])
        collinear_quad = np.array([
            [100.0, 100.0],
            [300.0, 200.0],
            [500.0, 300.0],
            [100.0, 400.0]
        ])
        swapped_quad = np.array([
            [100.0, 100.0],
            [500.0, 400.0],
            [500.0, 100.0],
            [100.0, 400.0]
        ])

        # 1. Three collinear points in source
        with self.assertRaises(homography.HomographyError) as ctx:
            homography.solve_homography(collinear_quad, valid_quad)
        msg_collinear_src = str(ctx.exception)
        self.assertIn("collinear", msg_collinear_src)
        self.assertIn("source", msg_collinear_src)

        # 2. Three collinear points in destination
        with self.assertRaises(homography.HomographyError) as ctx:
            homography.solve_homography(valid_quad, collinear_quad)
        msg_collinear_dst = str(ctx.exception)
        self.assertIn("collinear", msg_collinear_dst)
        self.assertIn("destination", msg_collinear_dst)

        # 3. Swapped corners (bow-tie) in source
        with self.assertRaises(homography.HomographyError) as ctx:
            homography.solve_homography(swapped_quad, valid_quad)
        msg_order_src = str(ctx.exception)
        self.assertIn("order", msg_order_src)
        self.assertIn("source", msg_order_src)

        # 4. Swapped corners (bow-tie) in destination
        with self.assertRaises(homography.HomographyError) as ctx:
            homography.solve_homography(valid_quad, swapped_quad)
        msg_order_dst = str(ctx.exception)
        self.assertIn("order", msg_order_dst)
        self.assertIn("destination", msg_order_dst)

        # 5. Horizon check helper directly raises with horizon message on mixed-sign w
        H_horiz = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.0, 1.0]
        ])
        src_horiz = np.array([
            [-200.0, 100.0],
            [300.0, 100.0],
            [300.0, 200.0],
            [-200.0, 200.0]
        ])
        with self.assertRaises(homography.HomographyError) as ctx:
            homography._check_horizon(H_horiz, src_horiz)
        msg_horizon = str(ctx.exception)
        self.assertIn("horizon", msg_horizon)

        # 6. Correspondences generated from such an H raise HomographyError end to end
        src_h = np.hstack([src_horiz, np.ones((4, 1))])
        dst_h = (H_horiz @ src_h.T).T
        dst_horiz = dst_h[:, :2] / dst_h[:, 2:3]
        with self.assertRaises(homography.HomographyError):
            homography.solve_homography(src_horiz, dst_horiz)

        # 7. Distinct messages for each failure type
        messages = {
            "collinear": msg_collinear_src,
            "order": msg_order_src,
            "horizon": msg_horizon
        }
        self.assertEqual(len(set(messages.values())), 3)


class TestBlendV2(unittest.TestCase):
    """V2.1-V2.7: feather weights and blending."""

    @staticmethod
    def _grid(h, w):
        return np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))

    def test_v21_weights_zero_outside_and_on_border(self):
        h, w = 60, 80
        x, y = self._grid(h, w)
        valid = np.ones((h, w), dtype=bool)
        valid[10:50, 10:70] = False
        wt = blend.edge_weights(x, y, valid, (h, w))
        # Zero on all four borders
        self.assertTrue(np.all(wt[0, :] == 0) and np.all(wt[-1, :] == 0))
        self.assertTrue(np.all(wt[:, 0] == 0) and np.all(wt[:, -1] == 0))
        # Zero outside the valid mask, even in the interior
        self.assertTrue(np.all(wt[10:50, 10:70] == 0))
        # Positive strictly inside
        self.assertGreater(wt[5, 5], 0)
        self.assertGreater(wt[55, 75], 0)
        self.assertGreater(wt[30, 5], 0)

    def test_v22_weights_sum_to_one_and_output_in_range(self):
        # Two identical constant images must reproduce the constant exactly:
        # only possible if normalized weights sum to 1 (M2) everywhere,
        # including the all-zero border fallback.
        h, w = 40, 50
        x, y = self._grid(h, w)
        valid = np.ones((h, w), dtype=bool)
        wt = blend.edge_weights(x, y, valid, (h, w))
        c = np.full((h, w), 100.0)
        out = blend.feather_blend([c, c], [wt, wt])
        self.assertTrue(np.allclose(out, 100.0))
        self.assertFalse(np.any(np.isnan(out)))

        # Differing constants: output stays inside [min, max] of inputs (M3 convexity)
        d = np.full((h, w), 140.0)
        out2 = blend.feather_blend([c, d], [wt, wt])
        self.assertTrue(np.all(out2 >= 100.0) and np.all(out2 <= 140.0))

    def test_v24_constant_images_blend_smoothly(self):
        h, w = 60, 100
        x, y = self._grid(h, w)
        valid_a = np.zeros((h, w), dtype=bool); valid_a[:, :70] = True
        valid_b = np.zeros((h, w), dtype=bool); valid_b[:, 30:] = True
        wa = blend.edge_weights(x, y, valid_a, (h, w))
        wb = blend.edge_weights(x, y, valid_b, (h, w))
        # Mimic warp_image output: zeros where invalid
        img_a = np.where(valid_a, 100.0, 0.0)
        img_b = np.where(valid_b, 140.0, 0.0)
        out = blend.feather_blend([img_a, img_b], [wa, wb])
        self.assertFalse(np.any(np.isnan(out)))
        # Non-overlap pixels are exact
        self.assertTrue(np.all(out[:, :30] == 100.0))
        self.assertTrue(np.all(out[:, 70:] == 140.0))
        # Overlap interior is strictly between the two constants
        mid = out[30, 45]
        self.assertGreater(mid, 100.0)
        self.assertLess(mid, 140.0)
        # Smooth: adjacent columns in the overlap never jump more than ~1 level
        col_diffs = np.abs(np.diff(out[30].astype(float)))
        self.assertLess(np.max(col_diffs[30:70]), 40.0)

    def test_v25_horizontal_pan_gives_linear_ramp(self):
        # A covers canvas cols [0,200), B covers [100,300): pure x-translation.
        # Image height 400 makes vertical border distances exceed the 100px
        # overlap on central rows, so weights reduce to the linear ramp (M4).
        h, w = 400, 300
        x, y = self._grid(h, w)
        valid_a = np.zeros((h, w), dtype=bool); valid_a[:, :200] = True
        valid_b = np.zeros((h, w), dtype=bool); valid_b[:, 100:] = True
        # Source coords: A is identity, B is shifted by -100 in x.
        xb = x - 100.0
        wa = blend.edge_weights(x, y, valid_a, (h, 200))
        wb = blend.edge_weights(xb, y, valid_b, (h, 200))
        img_a = np.where(valid_a, 0.0, 0.0)
        img_b = np.where(valid_b, 255.0, 0.0)
        out = blend.feather_blend([img_a, img_b], [wa, wb])
        # Central row: alpha_B must be the linear ramp (cx - 100) / 99.
        row = 200
        cols = np.arange(101, 199)
        expected = 255.0 * (cols - 100.0) / 99.0
        self.assertTrue(np.allclose(out[row, cols], expected, atol=2.0),
                        f"ramp mismatch: {out[row, cols[:5]]} vs {expected[:5]}")

    def test_v26_exposure_gain_recovers_brightness(self):
        h, w = 40, 50
        x, y = self._grid(h, w)
        valid = np.ones((h, w), dtype=bool)
        wt = blend.edge_weights(x, y, valid, (h, w))
        a = np.full((h, w), 100.0)
        b = np.full((h, w), 150.0)  # 1.5x brighter
        out_plain = blend.feather_blend([a, b], [wt, wt])
        self.assertTrue(np.allclose(out_plain[20, 25], 125.0, atol=1.0))
        out_gain = blend.feather_blend([a, b], [wt, wt], exposure_gain=True)
        self.assertTrue(np.allclose(out_gain, 100.0, atol=1.0),
                        f"gain recovery failed, center={out_gain[20, 25]}")

    def test_v27_seam_jump_at_least_50_percent_lower(self):
        # Same translation geometry as V2.5; constant offset of 100 between
        # images (exposure difference, no gain correction).
        h, w = 400, 300
        x, y = self._grid(h, w)
        valid_a = np.zeros((h, w), dtype=bool); valid_a[:, :200] = True
        valid_b = np.zeros((h, w), dtype=bool); valid_b[:, 100:] = True
        xb = x - 100.0
        wa = blend.edge_weights(x, y, valid_a, (h, 200))
        wb = blend.edge_weights(xb, y, valid_b, (h, 200))
        img_a = np.where(valid_a, 0.0, 0.0)
        img_b = np.where(valid_b, 100.0, 0.0)
        out = blend.feather_blend([img_a, img_b], [wa, wb])

        # Hard cut: pick the image with the larger weight per pixel.
        hard = np.where(wa >= wb, img_a, img_b)
        row = 200
        hard_jump = np.max(np.abs(np.diff(hard[row].astype(float))))
        feather_jump = np.max(np.abs(np.diff(out[row].astype(float))))
        self.assertGreater(hard_jump, 0)
        self.assertLessEqual(feather_jump, 0.5 * hard_jump)

    def test_valid_masks_fix_black_border_fallback(self):
        # Reviewer regression: a valid black first image at a zero-weight
        # border must not lose to a bright second image. With valid masks
        # passed, coverage — not pixel brightness — picks the hard-cut image.
        h, w = 60, 100
        x, y = self._grid(h, w)
        valid_a = np.zeros((h, w), dtype=bool); valid_a[:, :70] = True
        valid_b = np.zeros((h, w), dtype=bool); valid_b[:, 30:] = True
        wa = blend.edge_weights(x, y, valid_a, (h, w))
        wb = blend.edge_weights(x, y, valid_b, (h, w))
        img_a = np.where(valid_a, 0.0, 0.0)    # valid but black
        img_b = np.where(valid_b, 200.0, 0.0)
        out = blend.feather_blend([img_a, img_b], [wa, wb], valid=[valid_a, valid_b])
        # Top border inside the overlap: both weights are 0, A is first valid.
        self.assertEqual(out[0, 45], 0.0)
        # B-only region still picks B.
        self.assertEqual(out[30, 85], 200.0)

    def test_gain_output_clipped_to_dtype_range(self):
        # Reviewer regression: gain > 1 must not wrap bright uint8 pixels.
        # A=100 in overlap, B=50 in overlap and 200 in its own region -> g=2,
        # so B's lone region would hit 400 without clipping.
        h, w = 40, 50
        x, y = self._grid(h, w)
        valid_a = np.zeros((h, w), dtype=bool); valid_a[:, :30] = True
        valid_b = np.zeros((h, w), dtype=bool); valid_b[:, 20:] = True
        wa = blend.edge_weights(x, y, valid_a, (h, w))
        wb = blend.edge_weights(x, y, valid_b, (h, w))
        img_a = np.where(valid_a, 100, 0).astype(np.uint8)
        img_b = np.where(valid_b, np.where(x >= 30, 200, 50), 0).astype(np.uint8)
        out = blend.feather_blend([img_a, img_b], [wa, wb],
                                  exposure_gain=True, valid=[valid_a, valid_b])
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out[30, 45], 255)  # clipped, not wrapped to 144

    def test_zero_gain_is_applied(self):
        # Reviewer regression: a black reference overlap yields g = 0; the
        # gain must still be applied (B is driven to the reference value).
        h, w = 40, 50
        x, y = self._grid(h, w)
        valid = np.ones((h, w), dtype=bool)
        wt = blend.edge_weights(x, y, valid, (h, w))
        a = np.zeros((h, w))
        b = np.full((h, w), 100.0)
        out = blend.feather_blend([a, b], [wt, wt], exposure_gain=True)
        self.assertTrue(np.allclose(out[20, 25], 0.0, atol=1.0))

    def test_float16_gain_does_not_produce_inf(self):
        # Reviewer regression: gain-corrected float16 pixels above 65504
        # must not become inf; the result widens to float64.
        h, w = 40, 50
        x, y = self._grid(h, w)
        valid_a = np.zeros((h, w), dtype=bool); valid_a[:, :30] = True
        valid_b = np.zeros((h, w), dtype=bool); valid_b[:, 20:] = True
        wa = blend.edge_weights(x, y, valid_a, (h, w))
        wb = blend.edge_weights(x, y, valid_b, (h, w))
        img_a = np.where(valid_a, 60000, 0).astype(np.float16)
        img_b = np.where(valid_b, np.where(x >= 30, 50000, 40000), 0).astype(np.float16)
        out = blend.feather_blend([img_a, img_b], [wa, wb],
                                  exposure_gain=True, valid=[valid_a, valid_b])
        self.assertEqual(out.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(out)))
        # g ~= 1.5, so B's lone region lands near 75000 (float16 input
        # precision is ~32 at this magnitude).
        self.assertAlmostEqual(float(out[30, 45]), 75000.0, delta=500.0)


if __name__ == '__main__':
    unittest.main()

