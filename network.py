import tensorflow as tf

def dense_connection( input_layer, weight, bias ):
    image_layer = tf.add(tf.matmul(input_layer, weight), bias)
    output_layer = tf.nn.relu(features = image_layer )
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
        in_layer = tf.reshape(input_,[1,773])
        hidden_1 = encode(in_layer, "encode1", 773, 600)
        output_1 = decode(hidden_1, "decode1", 600, 773)
        ae1 = { 'hidden' : hidden_1 , 'output' : output_1, 'input':in_layer }
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
        
################################################################################

def auto_encoder_loss(model_function, level):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, shape=[773,])
            y_ = model_function(x_, level)['input']          
            y_coded = model_function(x_, level)['output']
            learn_rate = tf.placeholder(tf.float32, shape=[])
        
            losses = tf.nn.l2_loss(tf.subtract(y_coded, y_ ))
            
            with tf.name_scope("Squared-Error-Loss"):
                squared_error_loss = tf.reduce_mean(losses)
                tf.summary.scalar("loss", squared_error_loss )
            
            trainer = tf.train.AdamOptimizer(learning_rate = learn_rate)
            train_op = trainer.minimize(squared_error_loss)
            
            correct_prediction = tf.equal(y_coded, y_)
            with tf.name_scope('Accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
                tf.summary.scalar("accuracy", accuracy )            
            
            
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
    model_dict = {'graph': g, 'inputs': [x_] , 'rate':learn_rate, 'train_op': train_op,
                   'accuracy':accuracy, 'var_list':var_list, 'loss': squared_error_loss }
    
    return model_dict

################################################################################