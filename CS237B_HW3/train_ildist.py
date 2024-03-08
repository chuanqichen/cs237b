import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import argparse
from utils import *

tf.config.run_functions_eagerly(True)

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # IMPORTANT: out_size is still 2 in this case, because the action space is 2-dimensional. But your network will output some other size as it is outputing a distribution!
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        self.out_size = out_size
        l1_l2 = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.001)
        self.dense1 = tf.keras.layers.Dense(250, activation=tf.nn.leaky_relu, kernel_regularizer=l1_l2, bias_regularizer=l1_l2,kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.dense2 = tf.keras.layers.Dense(50, activation=tf.nn.leaky_relu, kernel_regularizer=l1_l2, bias_regularizer=l1_l2, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.dense3 = tf.keras.layers.Dense(out_size, kernel_regularizer=l1_l2, bias_regularizer=l1_l2, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.dense4 = tf.keras.layers.Dense(out_size*out_size, kernel_regularizer=l1_l2, bias_regularizer=l1_l2, kernel_initializer=tf.keras.initializers.GlorotUniform())

        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (?, |O|) tensor that keeps a batch of observations
        # IMPORTANT: First two columns of the output tensor must correspond to the mean vector!
        out = self.dense1(x)
        #out = self.dropout1(out)
        out = self.dense2(out)
        #out = self.dropout2(out)
        mu = self.dense3(out)        
        A = tf.reshape(self.dense4(out), [-1, self.out_size, self.out_size]) 
        covar = tf.einsum('nik, kjn->nij', A, tf.transpose(A))   # Sigma = A A.T => PSD 
        covar += tf.convert_to_tensor([[1e-3, 0], [0, 1e-3]])      # impove numerical stability 
        covar = tf.reshape(covar, [-1, 4])
        return tf.concat([mu, covar], -1)
        ########## Your code ends here ##########


   
def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the negative log-likelihood loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: You may find the classes of tensorflow_probability.distributions (imported as tfd) useful.
    #       In particular, you can use MultivariateNormalFullCovariance or MultivariateNormalTriL, but they are not the only way.
    mu = y_est[:, 0:2]
    covar = tf.reshape(y_est[:, 2:6], [-1, 2, 2])
    # mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covar)
    mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(covar))
    _loss  = -tf.reduce_mean(mvn.log_prob(y), axis=0)
    
    return _loss
    ########## Your code ends here ##########


def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096*32,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
       
        with tf.GradientTape() as tape:
            y_est = nn_model(x)
            current_loss = loss(y_est, y) #+  tf.add_n(nn_model.losses)
        grads = tape.gradient(current_loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))
              
        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)


    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=1e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
