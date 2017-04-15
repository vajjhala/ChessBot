import numpy as np
import gzip 
import random

def gzipper(matrix, file_name):
    with gzip.GzipFile("{0}.npy.gz".format(file_name), "w") as f:
        np.save(file=f, arr=matrix)
        
def rand_list(out_of, subset):
    list_ = []
    for i in range(subset):
        list_.append( random.randint(0, out_of) )
    list_b = list( set( range(out_of) ) - set(list_) )
    return list(set(list_)), list_b


def main( cross_val_num ):
    
    with gzip.GzipFile('CCRL_all.npy.gz', "r") as f:
        A = np.load(f)
     
    W = A[ A[:,-1] == 1 ]
    L = A[ A[:,-1] == 0 ]
    
    W_tuple = rand_list(W.shape[0], cross_val_num )
    L_tuple = rand_list(L.shape[0], cross_val_num )
    
    cv_W, train_W = W[W_tuple[0], :], W[W_tuple[1], :]
    cv_L, train_L = L[L_tuple[0], :], L[L_tuple[1], :]
  
    pairs = zip([cv_W, train_W, cv_L, train_L ], 
                ["cross_validation_wins", "train_wins", "cross_validation_loses","train_loses"])
    
    for pair in pairs:
        gzipper(pair[0],pair[1])
        
    print("Wins:", W.shape, "Loses:", L.shape)
    print( len(W_tuple[0]), len(W_tuple[1]), len(L_tuple[0]), len(L_tuple[1]) )
    print( cv_W.shape, train_W.shape, cv_L.shape, train_L.shape )

main(100000)