import tensorflow as tf
import numpy as np
import gzip


# creates our autoencoder
def autoencoder(x,hidden_sizes,init=tf.random_normal_initializer()):
    in_size = x.get_shape().as_list()
    in_size = in_size[-1]
    cur_size = in_size
    xlayer = x
    # do the encoding
    for i in range(len(hidden_sizes)):
        with tf.variable_scope('autoencoder%d' % i):
            W = tf.get_variable('encoderW',shape=(cur_size,hidden_sizes[i]),initializer=init)
            b = tf.get_variable('encoderb',shape=(hidden_sizes[i]),initializer=init)
            cur_size = hidden_sizes[i]
            xlayer = tf.nn.relu(tf.add(tf.matmul(tf.cast(xlayer,tf.float32),W),b))

    # do the decoding,
    xp = xlayer
    # end at i=1
    for i in range(len(hidden_sizes)-1,0,-1):
        with tf.variable_scope('autoencoder%d' % i):
            Wp = tf.get_variable('decoderW',shape=[hidden_sizes[i],hidden_sizes[i-1]],initializer=init)
            bp = tf.get_variable('decoderb',shape=[hidden_sizes[i-1]],initializer=init)
            xp = tf.nn.relu(tf.add(tf.matmul(tf.cast(xp,tf.float32),Wp),bp))
    # now back to the input size
    with tf.variable_scope('autoencoder0'):
        Wp = tf.get_variable('decoderW',shape=[hidden_sizes[0],in_size],initializer=init)
        bp = tf.get_variable('decoderb',shape=[in_size],initializer=init)
        xp = tf.nn.relu(tf.add(tf.matmul(tf.cast(xp,tf.float32),Wp),bp))
        return xp

# temporary to generate fake data
def generate_data(nObs):
    r = np.random.randint(0,high=2,size=(nObs,773))
    return r

def train_autoencoder(train_data,test_data,layer_sizes,learning_rate = 0.01):
    # build multi-layer autoencoder model
    my_train = {}
    my_losses = {}
    var_list = []
    with tf.Graph().as_default() as g, tf.variable_scope('dbn'):
        with tf.device("/cpu:0"):

            x_ = tf.placeholder(tf.int32, [None, 773])
            xin = x_
            xp = autoencoder(xin,layer_sizes)
            for i in range(len(layer_sizes)):
                cur_id = 'autoencoder%d'%i

                with tf.variable_scope(cur_id,reuse=True):
                    var_list.append(tf.get_variable('encoderW'))
                    var_list.append(tf.get_variable('encoderb'))
                # loss is the squared distance between x and its autoencodered self
                loss = tf.reduce_sum(tf.pow(tf.cast(xin,tf.float32)-xp,2),1)
                #losses = tf.reduce_mean(loss)
                my_losses[cur_id] = tf.reduce_mean(loss)
                # I heard somewhere this was a good Gradient Descent variant for Autoencoders
                # fix so only cur_ids variables can change
                scope = 'dbn/%s' % cur_id
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     )
                my_train[cur_id] = tf.train.RMSPropOptimizer(learning_rate).minimize(my_losses['autoencoder%d'%i],
                                                                                     var_list=train_vars)

    saver = tf.train.Saver(var_list=var_list)
    with g.as_default(), tf.Session() as sess, tf.variable_scope('dbn'):
        # write out graph
        writer = tf.summary.FileWriter('summaries/',sess.graph)
        sess.run(tf.global_variables_initializer())

        # so we want to train each layer in order
        print('Starting...')
        data = generate_data(100)
        for epoch_i in range(250):
            for i in range(len(layer_sizes)):
                my_id = 'autoencoder%d'%i
                _,l = sess.run([my_train[my_id],my_losses[my_id]],feed_dict={x_:train_data})
            ltrain = sess.run(my_losses['autoencoder0'],feed_dict={x_:train_data})
            ltest = sess.run(my_losses['autoencoder0'],feed_dict={x_:test_data})
            # write loss to a summary variable
            loss_train_summary = tf.Summary(value=[tf.Summary.Value(tag='autoencoder_loss_train',simple_value=ltrain)])
            loss_test_summary = tf.Summary(value=[tf.Summary.Value(tag='autoencoder_loss_train',simple_value=ltest)])
            writer.add_summary(loss_train_summary)
            writer.add_summary(loss_test_summary)
            print('Epoch %d: Train %f Test %f' % (epoch_i,ltrain,ltest))
            # adjust learning rate
            #learning_rate = learning_rate * 0.999
        saver.save(sess,'./autoencoder_weights')

def unpackData(data):
# so data is a numpy matrix of size (nObs,774)
    y = data[:,-1]
    x = data[:,:-1]
    wins = x[y==1,:]
    loss = x[y==0,:]
    return {'win':wins,'loss':loss,'all_data':x}


# size of the DBN (autoencoder)
# load in input data
print('Loading Data')
with gzip.GzipFile('Morphy.npy.gz', "r") as f:
   test_data = np.load(f).astype(dtype=np.int32)
with gzip.GzipFile('Sicilian.npy.gz', "r") as f:
   train_data= np.load(f).astype(dtype=np.int32)
print('Data loaded')

train_dat = unpackData(train_data)
test_dat= unpackData(test_data)

hidden_sizes = [600,400,200,100]
# train the autoencoder
train_autoencoder(train_dat['all_data'],test_dat['all_data'],hidden_sizes)
