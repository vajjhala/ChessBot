import gzip
import numpy as np

with gzip.GzipFile('CCRL.npy.gz', "r") as f:
    item = np.load(f)

print(item.shape)
y_ = item[:, -1]
print(np.unique(y_ , return_counts=True))
