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
    
<<<<<<< HEAD
@restartable
def auto_encoder_gen(batch_size):
    
=======
    
@restartable
def auto_encoder_gen(batch_size):

    ''' 
    Choose a random set of one million wins and loses of white
    and pass the instances without any information of the result 
    for feature extraction 
        
    Input : Give it the batch_size for training iteration in the autoencoder
        
    '''
        
>>>>>>> PGN_tester
    with gzip.GzipFile('train_wins.npy.gz', "r") as f:
        CCRL_wins = np.load(f)
        
    with gzip.GzipFile('train_loses.npy.gz', "r") as f:
        CCRL_loses = np.load(f)
<<<<<<< HEAD
        
    L_ = CCRL_loses[ :1000000,:-1] 
    W_  = CCRL_wins[:1000000,:-1]
    un_sup = np.concatenate([L_,W_], axis = 0)
    data_len = un_sup.shape[0]
    
=======
      
    # Ignoring the last coloumn which pertains to the result 
    L_ = CCRL_loses[ :1000000,:-1] # one million loses
    W_  = CCRL_wins[:1000000,:-1]  # One million wins
    
    # Join both and get a random shuffles set of 2 million instances
    un_sup = np.concatenate([L_,W_], axis = 0)  
    np.random.shuffle(un_sup)
    data_len = un_sup.shape[0]
    
    # Batch size for each training iteration 
>>>>>>> PGN_tester
    for slice_i in range(int(math.ceil(data_len / batch_size))):
        idx = slice_i * batch_size
        X_batch = un_sup[idx:idx + batch_size]
        yield X_batch.astype(np.int8)
   
@restartable
<<<<<<< HEAD
def gen_siemese(batch_size, data_type ):
=======
def siemese_generator(batch_size, data_type ):
>>>>>>> PGN_tester
    
    assert data_type in ['cross_validation', 'train']
    
    with gzip.GzipFile('{}_wins.npy.gz'.format(data_type), "r") as f:
        wins_ = np.load(f)
    with gzip.GzipFile('{}_loses.npy.gz'.format(data_type), "r") as f:
        loses_ = np.load(f)
    
<<<<<<< HEAD
=======
    ''' Training data generation  :
    
        1) Pick one million random instances from white wins and loses
        2) Concatenate them to create a matrix of 2 million instances
        3) Shuffle it to reate (W,L) or (L,W) pairs
        4) Sepearate the matrix into two parts 
        
        Send these two parts to the two branches of the siemese network
        as X1, X2
        
        Minize the loss with comparision to Y which is (1,0) or (0,1)
        which indicates which branch of the siemese was given the winning 
        position.
    '''
    
    # training data
>>>>>>> PGN_tester
    if data_type =='train':
        list_a = []
        list_b = []

        for x in range(1000000):
<<<<<<< HEAD
            list_a.append(random.randint(0,wins_.shape[0]))
            list_b.append(random.randint(0,loses_.shape[0]))
            
        index = math.ceil(len(list_a) / 2)
        L1, L2 = loses_[list_a][:index], loses_[list_a][index:]
        W1, W2 = wins_[list_b][:index], wins_[list_b][index:]
        
=======
            list_b.append(random.randint(0, (loses_.shape[0]-1) ) )
            list_a.append(random.randint(0,  (wins_.shape[0]-1) ) )
            
        index = math.ceil(len(list_a) / 2)
        L1, L2 = loses_[list_b][:index], loses_[list_b][index:]
        W1, W2 = wins_[list_a][:index], wins_[list_a][index:]
    
    # Cross validation data :
    # A set of about 10,000 instances of white wins and loses 
    # against which the model's accuracy will be compared
    
>>>>>>> PGN_tester
    else:
        index = math.ceil(loses_.shape[0] / 2)
        L1, L2 = loses_[:index], loses_[index:]
        W1, W2 = wins_[:index], wins_[index:]
    
    X_1 = np.concatenate((L1,W2), axis=0) 
    np.random.shuffle(X_1)
    X_2 = np.concatenate((L2,W1), axis=0)
    np.random.shuffle(X_2)

    X1 = X_1[:,:773]
    X2 = X_2[:,:773]
    Y = np.array(list(zip(X_1[:,-1], X_2[:,-1])))
<<<<<<< HEAD

=======
    
    # Mini batchs for gradient descent
>>>>>>> PGN_tester
    data_len = Y.shape[0]
    for slice_i in range(int(math.ceil(data_len / batch_size))):
        idx = slice_i * batch_size
        yield (X1[idx:idx + batch_size].astype(np.int8), X2[idx:idx + batch_size].astype(np.int8), 
                 Y[idx:idx + batch_size].astype(np.int8))
 
 
<<<<<<< HEAD
g = auto_encoder_gen(50000)
for iter_, batch in enumerate(g):
    print(batch.shape)
    
f = gen_siemese(500000, 'train')
for iter_,batch in enumerate(f):
    print(batch[0].shape, batch[1].shape, batch[2].shape)
=======
# g = auto_encoder_gen(50000)
# for iter_, batch in enumerate(g):
    # print(batch.shape)
    
# f = siemese_generator(50000, 'train')
# for iter_,batch in enumerate(f):
    # print(batch[0].shape, batch[1].shape, batch[2].shape)
>>>>>>> PGN_tester
