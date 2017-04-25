import numpy as np
import gzip

for key,x in enumerate(range(0,2)):
    with gzip.GzipFile('CCRL{}.npy.gz'.format(x), "r") as f:
        item = np.load(f)
        if key == 0:
            A = item
            print(A.shape,item.shape)
        else:
            A = np.concatenate((A,item))
            print(A.shape,item.shape)
            
print(A.shape)
y_ = A[:, -1:].reshape([-1])
print(np.unique(y_ , return_counts=True))