import numpy as np
import tensorflow as tf
import argparse
from utils import *

tf.config.run_functions_eagerly(True)

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the CoIL network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        l1_l2 = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.001)
        self.dense1 = tf.keras.layers.Dense(250, activation=tf.nn.leaky_relu, kernel_regularizer=l1_l2, bias_regularizer=l1_l2,kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.dense2s =[tf.keras.layers.Dense(50, activation=tf.nn.leaky_relu, kernel_regularizer=l1_l2, bias_regularizer=l1_l2, kernel_initializer=tf.keras.initializers.GlorotUniform())
                       for _ in range(3)]
                       
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.dense3s = [tf.keras.layers.Dense(out_size, kernel_regularizer=l1_l2, bias_regularizer=l1_l2, kernel_initializer=tf.keras.initializers.GlorotUniform())
                       for _ in range(3)]
        
        
        ########## Your code ends here ##########

    def call(self, x, u):
        x = tf.cast(x, dtype=tf.float32)
        u = tf.cast(u, dtype=tf.int8)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for (x,u) where:
        # - x is a (?, |O|) tensor that keeps a batch of observations
        # - u is a (?, 1) tensor (a vector indeed) that keeps the high-level commands (goals) to denote which branch of the network to use 
        # FYI: For the intersection scenario, u=0 means the goal is to turn left, u=1 straight, and u=2 right. 
        # HINT 1: Looping over all data samples may not be the most computationally efficient way of doing branching
        # HINT 2: While implementing this, we found tf.math.equal and tf.cast useful. This is not necessarily a requirement though.
        out = self.dense1(x)
        out = self.dropout1(out)
        goal_masks = tf.convert_to_tensor([tf.cast(tf.equal(u, i), tf.float32) for i in range(3)])
        out = goal_masks[0]*self.dense2s[0](out) + goal_masks[1]*self.dense2s[1](out) + goal_masks[2]*self.dense2s[2](out)
        out = self.dropout2(out)
        out = goal_masks[0]*self.dense3s[0](out) + goal_masks[1]*self.dense3s[1](out) + goal_masks[2]*self.dense3s[2](out)
        return out
        ########## Your code ends here ##########


def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations & goals,
    # - y is the actions the expert took for the corresponding batch of observations & goals
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally

    weight_steering = 10
    weight_throttle = 1.0
    _loss = tf.reduce_sum(tf.reduce_mean(tf.math.square(y_est - y), axis=0)*(tf.convert_to_tensor([weight_steering, weight_throttle])))

    return _loss; 
    ########## Your code ends here ##########
   

def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y, u):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model (note both x and u are inputs now)
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients

        with tf.GradientTape() as tape:
            y_est = nn_model(x, u)
            current_loss = loss(y_est, y) +  tf.add_n(nn_model.losses)
        grads = tape.gradient(current_loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))
        
        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y, u in train_data:
            train_step(x, y, u)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'], data['u_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    args.goal = 'all'
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
