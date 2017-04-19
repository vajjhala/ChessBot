import copy
import numpy as np
import random
import gzip
import math

class GeneratorRestartHandler(object):
    def __init__(self, gen_func, argv, kwargv):
        self.gen_func = gen_func
        self.argv = copy.copy(argv)
        self.kwargv = copy.copy(kwargv)
        self.local_copy = self.gen_func(*self.argv, **self.kwargv)
    
    def __iter__(self):
        return GeneratorRestartHandler(self.gen_func, self.argv, self.kwargv)
    
    def __next__(self):
        return next(self.local_copy)
    
    def next(self):
        return self.__next__()


def restartable(g_func):
    
    def tmp(*argv, **kwargv):
        return GeneratorRestartHandler(g_func, argv, kwargv)
    
    return tmp
    
    
@restartable
def auto_encoder_gen(batch_size):

    ''' 
    Choose a random set of one million wins and loses of white
    and pass the instances without any information of the result 
    for feature extraction 
        
    Input : Give it the batch_size for training iteration in the autoencoder
        
    '''
        
    with gzip.GzipFile('train_wins.npy.gz', "r") as f:
        CCRL_wins = np.load(f)
        
    with gzip.GzipFile('train_loses.npy.gz', "r") as f:
        CCRL_loses = np.load(f)
      
    # Ignoring the last coloumn which pertains to the result 
    L_ = CCRL_loses[ :1000000,:-1] # one million loses
    W_  = CCRL_wins[:1000000,:-1]  # One million wins
    
    # Join both and get a random shuffles set of 2 million instances
    un_sup = np.concatenate([L_,W_], axis = 0)  
    np.random.shuffle(un_sup)
    data_len = un_sup.shape[0]
    
    # Batch size for each training iteration 
    for slice_i in range(int(math.ceil(data_len / batch_size))):
        idx = slice_i * batch_size
        X_batch = un_sup[idx:idx + batch_size]
        yield X_batch.astype(np.int32)
   
@restartable
def siemese_generator(batch_size, data_type ):
    
    assert data_type in ['cross_validation', 'train']
    
    with gzip.GzipFile('{}_wins.npy.gz'.format(data_type), "r") as f:
        wins_ = np.load(f)
    with gzip.GzipFile('{}_loses.npy.gz'.format(data_type), "r") as f:
        loses_ = np.load(f)
    
    ''' Training data generation  :
    
        1) Pick one million random instances from white wins and loses
        2) Concatenate them to create a matrix of 2 million instances
        3) Shuffle it to create (W,L) or (L,W) pairs
        4) Separate the matrix into two parts 
        
        Send these two parts to the two branches of the siemese network
        as X1, X2
        
        Minize the loss with comparision to Y which is (1,0) or (0,1)
        which indicates which branch of the siemese was given the winning 
        position.
    '''
    
    # training data
    if data_type =='train':
    
        list_a = np.random.randint(0, wins_.shape[0], size =1000000 )
        list_b = np.random.randint(0, loses_.shape[0], size = 1000000 )

    # Cross validation data :
    # A set of about 10,000 instances of white wins and loses 
    # against which the model's accuracy will be compared
    
    else:
        list_a = np.random.randint(0, wins_.shape[0], size = 100000 )
        list_b = np.random.randint(0, loses_.shape[0], size = 100000 )
        
    index = math.ceil( len(list_a) / 2 )
    W1, W2 = wins_[list_a][:index], wins_[list_a][index:]
    L1, L2 = loses_[list_b][:index], loses_[list_b][index:]
    
    X_1 = np.concatenate((L1,W2), axis=0) 
    X_2 = np.concatenate((W1,L2), axis=0)

    assert X_1.shape[0] == X_2.shape[0]
    
    randomize = np.arange( X_1.shape[0] )
    np.random.shuffle(randomize)
    
    X1 = X_1[randomize][:,:773]
    X2 = X_2[randomize] [:,:773]
    
    Y = np.array(list(zip(X_1[:,-1], X_2[:,-1])))
    
    # Mini batchs for gradient descent
    data_len = Y.shape[0]
    for slice_i in range(int(math.ceil(data_len / batch_size))):
        idx = slice_i * batch_size
        yield (X1[idx:idx + batch_size].astype(np.int32), X2[idx:idx + batch_size].astype(np.int32), 
                 Y[idx:idx + batch_size].astype(np.int32))
 