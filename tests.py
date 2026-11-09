import os
import unittest
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

    def test_canvas_cap_extreme_h(self):
        H = np.array([[1.0e6, 0.0, 0.0], [0.0, 1.0e6, 0.0], [0.0, 0.0, 1.0]])
        canvas, _ = warp.canvas_bounds([H], [(100, 100)])
        cH, cW = canvas
        self.assertLessEqual(cH, warp.MAX_CANVAS_SIDE)
        self.assertLessEqual(cW, warp.MAX_CANVAS_SIDE)
        self.assertLessEqual(cH * cW, warp.MAX_CANVAS_PIXELS)

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


if __name__ == '__main__':
    unittest.main()

