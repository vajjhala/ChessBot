import tensorflow as tf

def dense_connection( input_layer, weight, bias ):
    ''' Affine Transformation followed by rectified linear units '''
    image_layer = tf.add(tf.matmul(input_layer, weight), bias)
    output_layer = tf.maximum(0.01*image_layer, image_layer)
    return output_layer
        
def encode(input_layer, scope_name, input_dimension, output_dimension ):

    with tf.variable_scope( str(scope_name) ) as scope :
        try:
            weight_ = tf.get_variable( name = "weights", 
                                       shape = [int(input_dimension), int(output_dimension)], 
                                       initializer = tf.random_normal_initializer() ) 
            
            bias_ = tf.get_variable( name = "bias", 
                                     shape = [ int(output_dimension) ],
                                     initializer = tf.constant_initializer(0.10) )

        except ValueError:
            scope.reuse_variables()
            weight_ = tf.get_variable("weights")
            bias_   = tf.get_variable("bias")

        z_ = dense_connection(input_layer, weight_ , bias_ )
        return z_

def decode( latent_layer, scope_name, input_dimension, output_dimension):

    with tf.variable_scope( str(scope_name) ) as scope :
        try:
            weight_ = tf.get_variable( name = "weights", 
                                       shape = [int(input_dimension), int(output_dimension)], 
                                       initializer = tf.random_normal_initializer() ) 
            
            bias_ = tf.get_variable( name = "bias", 
                                     shape = [ int(output_dimension) ],
                                     initializer = tf.constant_initializer(0.10) )

        except ValueError:
            scope.reuse_variables()
            weight_ = tf.get_variable("weights")
            bias_   = tf.get_variable("bias")

        output_ = dense_connection(latent_layer, weight_, bias_)
        return output_
        
def auto_encoder(input_, level):
    if level == 1:
        # Note : Shape for autoencoding is different!
        hidden_1 = encode(input_, "encode1", 773, 600)
        output_1 = decode(hidden_1, "decode1", 600, 773)
        ae1 = { 'hidden' : hidden_1 , 'output' : output_1, 'input':input_ } #input : in_layer
        return ae1

    if level == 2:
        in_layer = auto_encoder(input_, 1)['hidden']
        hidden_2 = encode(in_layer, "encode2", 600, 400)
        output_2 = decode(hidden_2, "decode2", 400, 600)
        ae2 = {'hidden' : hidden_2 , 'output' : output_2, 'input':in_layer}
        return ae2
    
    if level == 3:
        in_layer = auto_encoder(input_, 2)['hidden']
        hidden_3 = encode(in_layer, "encode3", 400, 200)
        output_3 = decode(hidden_3, "decode3", 200, 400)
        ae3 = {'hidden' : hidden_3 , 'output' : output_3, 'input':in_layer }
        return ae3
    
    if level == 4:
        in_layer = auto_encoder(input_, 3)['hidden']
        hidden_4 = encode(in_layer, "encode4", 200, 100)
        output_4 = decode(hidden_4, "decode4", 100, 200)
        ae4 = {'hidden' : hidden_4 , 'output' : output_4, 'input':in_layer }
        return ae4
 
def supervised_model( x1, x2 ):
    
    position_one = auto_encoder(x1, level = 4)['hidden']
    position_two = auto_encoder(x2, level = 4)['hidden']
    
    input_layer = tf.concat([position_one, position_two] , axis =1 )
 
    # Nothing is actually being encoded here but I am just using the same
    # function for feed forward.
    hidden_1 = encode( input_layer, "hidden_layer_one", 200, 400)
    hidden_2 = encode( hidden_1, "hidden_layer_two", 400, 200)
    hidden_3 = encode( hidden_2, "hidden_layer_three", 200, 100)
    
    with tf.variable_scope( "Final_layer" ) as scope :
    
        try:
            weight_ = tf.get_variable( name = "weights", 
                                       shape = [100, 2], 
                                       initializer = tf.random_normal_initializer() ) 
            
            bias_ = tf.get_variable( name = "bias", 
                                     shape = [ 2 ],
                                     initializer = tf.constant_initializer(0.10) )

        except ValueError:
            scope.reuse_variables()
            weight_ = tf.get_variable("weights")
            bias_   = tf.get_variable("bias")
    
        final_result = tf.add(tf.matmul( hidden_3, weight_ ), bias_ )
        
    return final_result

    
################################################################################

def auto_encoder_loss(model_function, level):
    with tf.Graph().as_default() as g:
        
    
        x_ = tf.placeholder(tf.float32, shape=[773])
        in_layer = tf.reshape(x_,[1,773]) 
        y_      = model_function(in_layer, level)['input']          
        y_coded = model_function(in_layer, level)['output']
        learn_rate = tf.placeholder(tf.float32, shape=[])
    
        losses = tf.nn.l2_loss(tf.subtract(y_coded, y_ ))
               
        with tf.name_scope("Squared-Error-Loss"):
            squared_error_loss = tf.reduce_mean(losses)
            tf.summary.scalar("loss", squared_error_loss )
        
        trainer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
        train_op = trainer.minimize(squared_error_loss)
        
        correct_prediction = tf.equal(y_coded, y_)
        with tf.name_scope('Accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
            tf.summary.scalar("accuracy", accuracy )            
        
        
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
    model_dict = {'graph': g, 'inputs': [x_] , 'rate':learn_rate, 'train_op': train_op,
                   'accuracy':accuracy, 'var_list':var_list, 'loss': squared_error_loss  }
    
    return model_dict

    
def supervised_loss( model_function ):
    
    with tf.Graph().as_default() as g:

        x1_ = tf.placeholder(tf.float32, shape=[None,773])
        x2_ = tf.placeholder(tf.float32, shape=[None,773])
        y_ =  tf.placeholder(tf.int32, [None,2])
        learn_rate = tf.placeholder(tf.float32, shape=[])
        
        y_logits = model_function(x1_, x2_)

        y_dict = dict( labels = y_, logits = y_logits )
        losses = tf.nn.softmax_cross_entropy_with_logits(**y_dict)
        
        with tf.name_scope('Supervised_Cross-Entropy-Loss'):
            cross_entropy_loss = tf.reduce_mean(losses)
            tf.summary.scalar("loss", cross_entropy_loss )
            
        trainer = tf.train.AdamOptimizer(learning_rate = learn_rate)
        train_op = trainer.minimize(cross_entropy_loss)
        
        
        correct_prediction = tf.equal(tf.argmax(y_logits,1), tf.argmax(y_,1))
        
        with tf.name_scope('Supervised_Accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
            tf.summary.scalar("accuracy", accuracy )            

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        auto_weights = []
        for scope_name in ['encode1', 'encode2', 'encode3', 'encode4'] :
            auto_weights.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= scope_name) )
        
    model_dict = {'graph': g, 'inputs': [x1_, x2_, y_ ] , 'rate':learn_rate, 'train_op': train_op,
                   'accuracy':accuracy, 'var_list':var_list, 'loss': cross_entropy_loss, 'encoder_vars' : auto_weights , 'output':y_logits}

    return model_dict
    
################################################################################