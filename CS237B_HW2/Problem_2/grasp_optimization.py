#!/usr/bin/env python

import cvxpy as cp
import numpy as np
import pdb  

from utils import *

def solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False):
    """
    Solves an SOCP of the form:

    minimize(h^T x)
    subject to:
        ||A_i x + b_i||_2 <= c_i^T x + d_i    for all i
        F x == g

    Args:
        x       - cvx variable.
        As      - list of A_i numpy matrices.
        bs      - list of b_i numpy vectors.
        cs      - list of c_i numpy vectors.
        ds      - list of d_i numpy vectors.
        F       - numpy matrix.
        g       - numpy vector.
        h       - numpy vector.
        verbose - whether to print verbose cvx output.

    Return:
        x - the optimal value as a numpy array, or None if the problem is
            infeasible or unbounded.
    """
    objective = cp.Minimize(h.T @ x)
    constraints = []
    for A, b, c, d in zip(As, bs, cs, ds):
        constraints.append(cp.SOC(c.T @ x + d, A @ x + b))
    constraints.append(F @ x == g)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    if prob.status in ['infeasible', 'unbounded']:
        return None

    return x.value

def grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext):
    """
    Solve the grasp force optimization problem as an SOCP. Handles 2D and 3D cases.

    Args:
        grasp_normals   - list of M surface normals at the contact points, pointing inwards.
        points          - list of M grasp points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).
        wrench_ext      - external wrench applied to the object.

    Return:
        f
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)
    transformations = [compute_local_transformation(n) for n in grasp_normals]

    ########## Your code starts here ##########
    n = D*M+1
    x = cp.Variable(n)
    h  = np.hstack([ np.zeros(n-1) , 1])
    g = -wrench_ext
    F = np.hstack([   
        np.vstack([transformations[i], cross_matrix(points[i]).dot(transformations[i]) ])  for i in range(M)
    ])
    F = np.hstack((F,np.zeros([F.shape[0],1], F.dtype)))

    As = []
    bs = []
    cs = []
    ds = []

    # rewrite the objective to SOCP with constraints part 
    for i in range(M):
        A = np.zeros([D, n])
        A[0:D, i*D:(i+1)*D] = np.identity(D)
        As.append(A)
        bs.append(np.zeros(D))
        c = np.hstack([np.zeros(n-1), 1])
        cs.append(c)
        ds.append(0)

    # rewrite friction cone constraints
    for i in range(M):
        A = np.zeros([D, n])
        Ai = np.identity(D)
        Ai[D-1, D-1] = 0
        A[0:D, i*D:(i+1)*D] = Ai
        As.append(A)
        bs.append(np.zeros(D))
        c = np.zeros(n)
        c[(i+1)*D-1] = friction_coeffs[i]
        cs.append(c)
        ds.append(0)

    x = solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False)

    # TODO: extract the grasp forces from x as a stacked 1D vector
    f = x[0:n-1]  # remove the last design variable s 
    ########## Your code ends here ##########

    # Transform the forces to the global frame
    F = f.reshape(M,D)
    forces = [T.dot(f) for T, f in zip(transformations, F)]

    return forces

def precompute_force_closure(grasp_normals, points, friction_coeffs):
    """
    Precompute the force optimization problem so that force closure grasps can
    be found for any arbitrary external wrench without redoing the optimization.

    Args:
        grasp_normals   - list of M contact normals, pointing inwards from the object surface.
        points          - list of M contact points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).

    Return:
        force_closure(wrench_ext) - a function that takes as input an external wrench and
                                    returns a set of forces that maintains force closure.
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)

    ########## Your code starts here ##########
    # TODO: Precompute the optimal forces for the 12 signed unit external
    #       wrenches and store them as rows in the matrix F. This matrix will be
    #       captured by the returned force_closure() function.
    F = np.zeros((2*N, M*D))

    for i in range(N):
        wrench_ext_pos = np.zeros(N)
        wrench_ext_pos[i] = 1
        F[i*2, :] = np.array(grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext_pos)).reshape(-1)
        wrench_ext_neg = np.zeros(N)
        wrench_ext_neg[i] = -1
        F[i*2+1, :] = np.array(grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext_neg)).reshape(-1)

    ########## Your code ends here ##########

    def force_closure(wrench_ext):
        """
        Return a set of forces that maintain force closure for the given
        external wrench using the precomputed parameters.

        Args:
            wrench_ext - external wrench applied to the object.

        Return:
            forces - grasp forces as a list of M numpy arrays.
        """

        ########## Your code starts here ##########
        # TODO: Compute the force closure forces as a stacked vector of shape (M*D)
        f = np.zeros(M*D)
        for i in range(N):
            wrench_pos = max(0, wrench_ext[i])
            wrench_neg = max(0, -wrench_ext[i])
            f += wrench_pos*F[2*i];
            f += wrench_neg*F[2*i+1];
        ########## Your code ends here ##########

        forces = [f_i for f_i in f.reshape(M,D)]
        return forces

    return force_closure
