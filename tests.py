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


if __name__ == '__main__':
    unittest.main()

