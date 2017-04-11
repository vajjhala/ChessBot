import tensorflow as tf
import numpy as np
import pickle


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

def train_autoencoder(in_data,layer_sizes,learning_rate = 0.01):
    #input_data = generate_data(5)
    # build multi-layer autoencoder model
    my_train = {}
    my_losses = {}
    var_list = []
    with tf.Graph().as_default() as g:
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
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     cur_id)
                my_train[cur_id] = tf.train.RMSPropOptimizer(learning_rate).minimize(my_losses['autoencoder%d'%i],
                                                                                     var_list=train_vars)

    saver = tf.train.Saver(var_list=var_list)
    with g.as_default(), tf.Session() as sess:
        # write out graph
        writer = tf.summary.FileWriter('summaries/',sess.graph)
        sess.run(tf.global_variables_initializer())

        # so we want to train each layer in order
        print('Starting...')
        data = generate_data(100)
        for epoch_i in range(250):
            for i in range(len(layer_sizes)):
                my_id = 'autoencoder%d'%i
                _,l = sess.run([my_train[my_id],my_losses[my_id]],feed_dict={x_:in_data})
            l = sess.run(my_losses['autoencoder0'],feed_dict={x_:in_data})
            # write loss to a summary variable
            loss_summary = tf.Summary(value=[tf.Summary.Value(tag='autoencoder_loss',simple_value=l)])
            writer.add_summary(loss_summary)
            print('Epoch %d: %f (LR = %f)' % (epoch_i,l,learning_rate))
            # adjust learning rate
            learning_rate = learning_rate * 0.999
        saver.save(sess,'./autoencoder_weights')


# size of the DBN (autoencoder)
hidden_sizes = [600,400,200,100]
# load in input data
with open('Morphy.pickle','rb') as file:
    input_data = pickle.load(file)
# train the autoencoder
train_autoencoder(input_data,hidden_sizes)
