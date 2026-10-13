import numpy as np
from pydmd import HODMD

# Make a random testmatrix (2 rows (channels), 100 timestamps)
fake_data = np.random.rand(2, 100)

hodmd = HODMD(svd_rank=0, d=3)

hodmd.fit(fake_data)

print(f"modes shape: {hodmd.modes.shape}")