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


if __name__ == '__main__':
    unittest.main()

