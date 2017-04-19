import tensorflow as tf
import numpy as np
import read_data
import network
import sys

def encoder_trainer(model_dict, dataset_generators, epoch_n, print_every, model_path, rate, decay, load_model=True,
                    save_model=True):

    log_dir = './tmp/tb_events'

    with model_dict['graph'].as_default(), tf.Session(config=tf.ConfigProto( allow_soft_placement=True,log_device_placement=True)) as sess:
        
        sess.run(tf.global_variables_initializer())
        
        if load_model == True:
            print("----------Restoring--------")
            saver = tf.train.Saver( var_list = model_dict['var_list'] )
            saver.restore(sess, model_path) 
            # restore automatically initialises

        train_writer = tf.summary.FileWriter(log_dir + '/auto_encoder', sess.graph)
        merged = tf.summary.merge_all()
         
        print(model_dict['var_list'])
        
        for epoch_i in range(epoch_n):
            cur_rate =  rate * ( decay ** epoch_i)
            for iter_i, data_batch in enumerate(dataset_generators['train']):
            
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                train_feed_dict[model_dict['rate']] = cur_rate
                _, summary = sess.run([ model_dict['train_op'], merged ], feed_dict=train_feed_dict)
                
                
                if iter_i % print_every == 0:
                
                    train_writer.add_summary(summary)
                    collect_arr = []
                    to_compute = [model_dict['loss'], model_dict['accuracy']]
                    collect_arr.append( sess.run( to_compute, train_feed_dict ) ) 
                    
                    averages = np.mean(collect_arr, axis=0)
                    avg_tpl = tuple(averages)
                    fmt = (epoch_i, iter_i, ) + avg_tpl
                    
                    print('iteration {:d} {:d}\t loss: {:.3f} ,'
                        'accuracy: {:.3f}'.format(*fmt))
        
        train_writer.close()
        
        if save_model == True:
            print("-----------Saving ------")
            saver = tf.train.Saver( var_list = model_dict['var_list'] )
            save_path = saver.save(sess, model_path)
            print ("Model saved in file: %s" % save_path)
   
###############################################################################

def auto_encoder_train():

    dataset_generators = { 'train': read_data.auto_encoder_gen(2500) }
    
    print("-------------------Layer 1------------------------------")  
    
    model_dict1 = network.auto_encoder_loss(network.auto_encoder, level = 1)
    encoder_trainer(model_dict1, dataset_generators, epoch_n=200, print_every=100, 
                    rate= 0.005, decay= 0.98, model_path = './tmp/encoder_ae1.ckpt')
    print("-------------------Layer 2------------------------------")   
    
    model_dict1 = network.auto_encoder_loss(network.auto_encoder, level = 2)
    encoder_trainer(model_dict1, dataset_generators, epoch_n=200, print_every=100, 
                    rate= 0.005, decay= 0.98, model_path="./tmp/encoder_ae2.ckpt")
    
    print("-------------------Layer 3------------------------------")  
    
    model_dict1 = network.auto_encoder_loss(network.auto_encoder, level = 3)
    encoder_trainer(model_dict1, dataset_generators, epoch_n=200, print_every=100, 
                    rate= 0.005, decay= 0.98, model_path="./tmp/encoder_ae3.ckpt")
                    
    print("-------------------Layer 4------------------------------")      
    
    model_dict1 = network.auto_encoder_loss(network.auto_encoder, level = 4)
    encoder_trainer(model_dict1, dataset_generators, epoch_n=200, print_every=100, 
                    rate= 0.005, decay= 0.98, model_path="./tmp/encoder_ae4.ckpt")     

                    
auto_encoder_train()

###############################################################################

