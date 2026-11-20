./venv/bin/python demo.py rectify_panorama data/hero/starry_with_corners.png data/hero/starry_right.png data/hero/starry_pairs.JSON data/hero/starry_corners.JSON left 4:3

./venv/bin/python demo.py rectify_panorama data/hero/mario_with_corners.png data/hero/mario_right.png data/hero/mario_pairs.JSON data/hero/mario_corners.JSON left 4:3

./venv/bin/python demo.py rectify data/hero/starry_with_corners.png data/hero/starry_corners.JSON 4:3

./venv/bin/python demo.py rectify data/hero/mario_with_corners.png data/hero/mario_corners.JSON 4:3

./venv/bin/python demo.py panorama data/hero/starry_with_corners.png data/hero/starry_right.png data/hero/starry_pairs.JSON

./venv/bin/python run_demo_with_cv.py rectify_panorama inputs/starry_with_corners.png inputs/starry_right.png left 4:3

./venv/bin/python run_demo_with_cv.py rectify_panorama inputs/mario_with_corners.png inputs/mario_right.png left 4:3

./venv/bin/python run_demo_with_cv.py rectify inputs/starry_with_corners.png 4:3

./venv/bin/python run_demo_with_cv.py rectify inputs/mario_with_corners.png 4:3

./venv/bin/python run_demo_with_cv.py panorama inputs/starry_with_corners.png inputs/starry_right.png
