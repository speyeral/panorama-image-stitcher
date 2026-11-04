## Rules 

1. **Only the Lead merges.** A PR is ready when its description lists the V-IDs it covers, those tests are in `tests.py`, `python tests.py` passes, and its signatures match the contract.
2. **PRs are due by noon** on the deadline day.
3. **Linear algebra only through `linalg.py`.** A test fails if np.linalg appears outside `linalg.py` and `tests.py`. After Person 4’s swap, `linalg.py` loses its exemption too.
4. **Lead’s check per PR (about 10 minutes):** run `python tests.py`; confirm the V-IDs listed are really tested; confirm signatures; for image work, run the demo or look at the output; merge.
5. **Merge order, Nov 9:** Person 4’s `linalg.py` core (only if V0.1 and V0.2 pass, else the wrapper stays), then Person 3, then Person 2, then your own `homography.py`. **Nov 11:** Person 4’s remaining `linalg.py`, re-run all tests, then your LSQ, then blend, then the panorama demo.
6. **Paper caveats to carry into every math note:** the normal equations come from calculus or Chapter 5 (outside Chapters 1–4); the claim that AᵀA squares the condition number needs singular values (outside our topics); four points in general position giving rank 8 is stated as a standard fact and verified numerically; the one-dimensional null space is generic.
