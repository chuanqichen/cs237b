import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from utils import generate_problem, visualize_value_function


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    # perform value iteration
    for _ in range(1000):
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition
        V_prev = V
        Q = reward + gam*tf.einsum('ans,s->na', tf.convert_to_tensor(Ts), V)
        terminal_masks = tf.tile(tf.reshape(tf.cast(terminal_mask, bool), [sdim, 1]), [1, adim])
        V = tf.reduce_max(tf.where(terminal_masks, reward, Q), axis=1) # reduce by maxing over action
        err = tf.linalg.norm(V - V_prev)
        ######### Your code ends here ###########

        if err < 1e-7:
            break

    return V

def calculate_optimal_policy(problem, reward, gam, V_opt):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert reward.ndim == 2
    Q = reward + gam*tf.einsum('ans,s->na', tf.convert_to_tensor(Ts), V_opt)
    Policy_opt = tf.argmax(Q, axis=1)
    return Policy_opt

def visualize_policy_function(V, Policy):
    """
    Visualizes the value function given in V & computes the optimal action,
    visualized as an arrow.

    You need to call plt.show() yourself.

    Args:
        V: (np.array) the value function reshaped into a 2D array.
    """
    V = np.array(V)
    assert V.ndim == 2
    m, n = V.shape
    pos2idx = np.arange(m * n).reshape((m, n))
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    u, v = [], []
    for pt in pts:
        pt_min, pt_max = [0, 0], [m - 1, n - 1]
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_right, pt_up, pt_left, pt_down]
        idx = Policy[pt[0], pt[1]]
        #Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        #idx = np.argmax(Vs)
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])
    u, v = np.reshape(u, (m, n)), np.reshape(v, (m, n))

    plt.imshow(V.T, origin="lower")
    plt.quiver(X, Y, u, v, pivot="middle")



# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    visualize_value_function(np.array(V_opt).reshape((n, n)))
    plt.title("value iteration")
    plt.show()

    Policy_opt = calculate_optimal_policy(problem, reward, gam, V_opt)
    plt.figure(215)
    visualize_policy_function(np.array(V_opt).reshape((n, n)), np.array(Policy_opt).reshape((n, n)))
    plt.title("Optimal Policy")
    plt.show()


if __name__ == "__main__":
    main()
