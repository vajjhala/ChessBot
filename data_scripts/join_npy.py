import numpy as np
import gzip

for key,x in enumerate(range(0,16)):
    with gzip.GzipFile('CCRL{}.npy.gz'.format(x), "r") as f:
        item = np.load(f)
        if key == 0:
            A = item
            print(A.shape,"CCRL{}shape".format(x),item.shape)
        else:
            A = np.concatenate((A,item))
            print(A.shape,"CCRL{}shape".format(x), item.shape)
            
with gzip.GzipFile('CCRL_all.npy.gz', "w") as f:
    np.save(file=f, arr=A)
