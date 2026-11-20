./venv/bin/python demo.py rectify_panorama data/hero/starry_with_corners.png data/hero/starry_right.png data/hero/starry_pairs.JSON data/hero/starry_corners.JSON left 4:3

./venv/bin/python demo.py rectify_panorama data/hero/mario_with_corners.png data/hero/mario_right.png data/hero/mario_pairs.JSON data/hero/mario_corners.JSON left 4:3

./venv/bin/python demo.py rectify data/hero/starry_with_corners.png data/hero/starry_corners.JSON 4:3

./venv/bin/python demo.py rectify data/hero/mario_with_corners.png data/hero/mario_corners.JSON 4:3

./venv/bin/python demo.py panorama data/hero/starry_with_corners.png data/hero/starry_right.png data/hero/starry_pairs.JSON 

./venv/bin/python demo.py rectify data/hero/squares_left.png data/hero/squares_corners.JSON 4:3

./venv/bin/python demo.py panorama data/hero/squares_left.png data/hero/squares_right.png data/hero/squares_pairs.JSON 

./venv/bin/python demo.py panorama data/hero/level_left.png data/hero/level_right.png data/hero/level_pairs.JSON 

./venv/bin/python demo.py rectify_panorama data/hero/level_left.png data/hero/level_right.png data/hero/level_pairs.JSON data/hero/level_corners.JSON left 4:3

./venv/bin/python demo.py match data/hero/level_left.png data/hero/level_right.png data/hero/level_pairs.JSON 
