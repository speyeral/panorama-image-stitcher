import os
import unittest
import io as io_std
from contextlib import redirect_stdout
import numpy as np

import linalg
import homography
import warp

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


class TestLinalgAcceptance(unittest.TestCase):
    def test_v01_rref_is_independent_of_row_order(self):
        rng = np.random.default_rng(20261111)
        for _ in range(100):
            full_rank = rng.normal(size=(8, 8))
            rank_deficient = rng.integers(-5, 6, size=(8, 3)) @ rng.integers(-5, 6, size=(3, 8))
            for matrix in (full_rank, rank_deficient):
                expected = linalg.rref(matrix)
                permuted = matrix[rng.permutation(matrix.shape[0])]
                actual = linalg.rref(permuted)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)

    def test_v02_solve_and_inverse_match_numpy(self):
        rng = np.random.default_rng(20261112)
        for _ in range(200):
            A = rng.normal(size=(8, 8))
            b = rng.normal(size=8)
            actual_x = linalg.solve(A, b)
            expected_x = np.linalg.solve(A, b)
            self.assertLessEqual(
                np.linalg.norm(actual_x - expected_x) / np.linalg.norm(expected_x),
                1e-8,
            )

            actual_inverse = linalg.inverse(A)
            expected_inverse = np.linalg.inv(A)
            self.assertLessEqual(
                np.linalg.norm(actual_inverse - expected_inverse) / np.linalg.norm(expected_inverse),
                1e-8,
            )
            np.testing.assert_allclose(A @ actual_inverse, np.eye(8), rtol=0.0, atol=1e-10)

        singular = np.array([[1.0, 2.0], [2.0, 4.0]])
        with self.assertRaises(linalg.SingularMatrixError):
            linalg.solve(singular, [1.0, 2.0])
        with self.assertRaises(linalg.SingularMatrixError):
            linalg.inverse(singular)

    def test_v03_linear_algebra_identities(self):
        rng = np.random.default_rng(20261113)
        A = rng.normal(size=(4, 4))
        B = rng.normal(size=(4, 4))
        self.assertAlmostEqual(linalg.det(A @ B), linalg.det(A) * linalg.det(B), delta=1e-8)
        self.assertAlmostEqual(linalg.det(A.T), linalg.det(A), delta=1e-8)
        swapped = A.copy()
        swapped[[0, 1]] = swapped[[1, 0]]
        self.assertAlmostEqual(linalg.det(swapped), -linalg.det(A), delta=1e-8)

        rank_three = rng.normal(size=(4, 3)) @ rng.normal(size=(3, 5))
        self.assertEqual(linalg.rank(rank_three) + (5 - linalg.rank(rank_three)), 5)
        self.assertLessEqual(
            np.linalg.norm(rank_three @ linalg.nullspace_vector(rank_three)),
            1e-9,
        )

        invertible = np.array([[2.0, -1.0, 3.0], [1.0, 4.0, 2.0], [3.0, 1.0, 5.0]])
        cofactors = np.empty((3, 3))
        for row in range(3):
            for col in range(3):
                minor = np.delete(np.delete(invertible, row, axis=0), col, axis=1)
                cofactors[row, col] = (-1) ** (row + col) * linalg.det(minor)
        adjugate = cofactors.T
        np.testing.assert_allclose(
            linalg.inverse(invertible),
            adjugate / linalg.det(invertible),
            rtol=0.0,
            atol=1e-10,
        )

        rhs = np.array([3.0, 7.0, 2.0])
        determinant = linalg.det(invertible)
        cramer_solution = []
        for col in range(3):
            replaced = invertible.copy()
            replaced[:, col] = rhs
            cramer_solution.append(linalg.det(replaced) / determinant)
        np.testing.assert_allclose(
            linalg.solve(invertible, rhs),
            cramer_solution,
            rtol=0.0,
            atol=1e-10,
        )

        consistent_A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        consistent_b = np.array([3.0, 6.0, 9.0])
        inconsistent_b = np.array([3.0, 6.0, 10.0])
        self.assertEqual(
            linalg.rank(consistent_A),
            linalg.rank(np.column_stack((consistent_A, consistent_b))),
        )
        self.assertNotEqual(
            linalg.rank(consistent_A),
            linalg.rank(np.column_stack((consistent_A, inconsistent_b))),
        )


class TestLeastSquaresHomography(unittest.TestCase):
    def _project(self, H, points):
        points_h = np.column_stack((points, np.ones(len(points))))
        mapped = (H @ points_h.T).T
        return mapped[:, :2] / mapped[:, 2, None]

    def test_v11_noise_free_eight_and_twelve_points(self):
        rng = np.random.default_rng(20261114)
        H_true = np.array([
            [1.08, 0.03, 42.0],
            [-0.02, 0.94, 31.0],
            [0.00008, -0.00004, 1.0],
        ])
        for count in (8, 12):
            src = rng.uniform(0.0, 800.0, size=(count, 2))
            dst = self._project(H_true, src)
            estimated = homography.solve_homography_lsq(src, dst)
            np.testing.assert_allclose(estimated, H_true, rtol=0.0, atol=1e-8)
            self.assertLess(homography.algebraic_residual(estimated, src, dst), 1e-16)

    def test_v12_v14_normal_equation_optimality_and_four_point_comparison(self):
        rng = np.random.default_rng(20261115)
        H_true = np.array([
            [1.03, -0.04, 18.0],
            [0.05, 0.97, -23.0],
            [0.00005, 0.00002, 1.0],
        ])
        rectangle = np.array([[0.0, 0.0], [600.0, 0.0], [600.0, 450.0], [0.0, 450.0]])
        additional = rng.uniform([0.0, 0.0], [600.0, 450.0], size=(8, 2))
        src = np.vstack((rectangle, additional))
        dst = self._project(H_true, src) + rng.normal(0.0, 0.15, size=(len(src), 2))

        estimated = homography.solve_homography_lsq(src, dst)
        residual = homography.algebraic_residual(estimated, src, dst)
        src_norm, _ = homography.normalize_points(src)
        dst_norm, _ = homography.normalize_points(dst)
        A, b = homography.build_system(src_norm, dst_norm)
        H_norm = _normalized_homography(estimated, src, dst)
        h = np.array([
            H_norm[0, 0], H_norm[0, 1], H_norm[0, 2],
            H_norm[1, 0], H_norm[1, 1], H_norm[1, 2],
            H_norm[2, 0], H_norm[2, 1],
        ])
        normal_residual = A.T @ (A @ h - b)
        self.assertLess(
            np.linalg.norm(normal_residual) / max(np.linalg.norm(A.T @ b), 1.0),
            1e-8,
        )

        for _ in range(1000):
            perturbation = rng.normal(size=8)
            perturbation *= 1e-5 / max(np.linalg.norm(perturbation), 1e-15)
            candidate = h + perturbation
            candidate_error = np.linalg.norm(A @ candidate - b) ** 2
            self.assertGreaterEqual(candidate_error + 1e-12, residual)

        exact_four_point = homography.solve_homography(src[:4], dst[:4])
        four_point_residual = homography.algebraic_residual(exact_four_point, src, dst)
        self.assertLessEqual(residual, four_point_residual + 1e-12)

    def test_v15_rank_of_normal_matrix_matches_design_matrix(self):
        rng = np.random.default_rng(20261116)
        src = rng.uniform(0.0, 500.0, size=(12, 2))
        dst = self._project(np.array([
            [1.02, 0.01, 12.0],
            [-0.01, 0.98, 15.0],
            [0.00002, -0.00003, 1.0],
        ]), src)
        src_norm, _ = homography.normalize_points(src)
        dst_norm, _ = homography.normalize_points(dst)
        A, _ = homography.build_system(src_norm, dst_norm)
        self.assertEqual(linalg.rank(A.T @ A), linalg.rank(A))

    def test_v16_collinear_points_raise(self):
        src = np.column_stack((np.arange(8.0), 2.0 * np.arange(8.0)))
        dst = src + np.array([10.0, -3.0])
        with self.assertRaises(homography.HomographyError):
            homography.solve_homography_lsq(src, dst)


def _normalized_homography(H, src, dst):
    _, T_src = homography.normalize_points(src)
    _, T_dst = homography.normalize_points(dst)
    result = T_dst @ H @ linalg.inverse(T_src)
    return result / result[2, 2]


if __name__ == '__main__':
    unittest.main()
