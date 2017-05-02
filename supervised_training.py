import tensorflow as tf
import numpy as np
import read_data
import network
import sys

def supervised_trainer(model_dict, epoch_n, print_every, model_path, rate, decay, load_model=False, save_model=True):

    log_dir = './tmp/tb_events'

    with  model_dict['graph'].as_default(), tf.Session(config=tf.ConfigProto( allow_soft_placement=True,log_device_placement=True)) as sess:
        
        sess.run(tf.global_variables_initializer())

        if load_model == True:
            print("----------Restoring--------")
            saver = tf.train.Saver( var_list = model_dict['var_list'] )
            saver.restore(sess, model_path) 
            # restore automatically initialises
                
        for level, ae_vars in enumerate(model_dict['encoder_vars']):
            ae_saver = tf.train.Saver( var_list = ae_vars  )
            ae_saver.restore(sess, './tmp/encoder_ae{}.ckpt'.format(level+1) )
            
        test_writer = tf.summary.FileWriter(log_dir + '/feed_forward_test', sess.graph) 
        train_writer = tf.summary.FileWriter(log_dir + '/feed_forward_train', sess.graph) 
        merged = tf.summary.merge_all()
        
        for epoch_i in range(epoch_n):
        
            
            dataset_generators = { 'train' : read_data.siemese_generator(2500, 'train'),
                                    'test' : read_data.siemese_generator(1000, 'cross_validation') }
                                    
            cur_rate =  rate * ( decay ** epoch_i)     
            
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                train_feed_dict[model_dict['rate']] = cur_rate
                _, summary = sess.run([ model_dict['train_op'], merged ], feed_dict=train_feed_dict)
                train_writer.add_summary(summary)
                #print( "Y_logits", sess.run( model_dict['output'] , feed_dict=train_feed_dict ) )
                #print(" Y_actual", data_batch[2] )
                if iter_i % print_every == 0:
                    collect_arr = []
                    
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict)) 
                        
                        test_writer.add_summary(sess.run(merged, test_feed_dict) )
                        
                    averages = np.mean(collect_arr, axis=0)
                    avg_tpl = tuple(averages)
                    fmt = (epoch_i, iter_i, ) + avg_tpl
                    
                    print('iteration {:d} {:d}\t loss: {:.3f} ,'
                        'accuracy: {:.3f}'.format(*fmt))
        
        test_writer.close()
        
        if save_model == True:
            print("-----------Saving ------")
            saver = tf.train.Saver( var_list = model_dict['var_list'] )
            save_path = saver.save(sess, model_path)
            print ("Model saved in file: %s" % save_path)
                 
###############################################################################
                    
def chess_learning():
                            
    model_dictionary = network.supervised_loss( network.supervised_model )
    
    supervised_trainer( model_dictionary, epoch_n=1000, print_every= 50, 
                        rate=0.005 , decay=0.90 , model_path= "./tmp/feed_forward.ckpt" )
    
chess_learning()
###############################################################################
