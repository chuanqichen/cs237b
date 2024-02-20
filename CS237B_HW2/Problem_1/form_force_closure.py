import cvxpy as cp
import numpy as np

def cross_matrix(x):
    """
    Returns a matrix x_cross such that x_cross.dot(y) represents the cross
    product between x and y.

    For 3D vectors, x_cross is a 3x3 skew-symmetric matrix. For 2D vectors,
    x_cross is a 2x1 vector representing the magnitude of the cross product in
    the z direction.
     """
    D = x.shape[0]
    if D == 2:
        return np.array([[-x[1], x[0]]])
    elif D == 3:
        return np.array([[0., -x[2], x[1]],
                         [x[2], 0., -x[0]],
                         [-x[1], x[0], 0.]])
    raise RuntimeError("cross_matrix(): x must be 2D or 3D. Received a {}D vector.".format(D))

def wrench(f, p):
    """
    Computes the wrench from the given force f applied at the given point p.
    Works for 2D and 3D.

    Args:
        f - 2D or 3D contact force.
        p - 2D or 3D contact point.

    Return:
        w - 3D or 6D contact wrench represented as (force, torque).    
    """
    ########## Your code starts here ##########
    # Hint: you may find cross_matrix(x) defined above helpful. This should be one line of code.
    D = f.shape[0]
    if D==2:
        w = np.hstack([f, cross_matrix(p)[0].dot(f)])
    else:
        w = np.hstack([f, cross_matrix(p).dot(f)])
    ########## Your code ends here ##########
    return w

def cone_edges(f, mu):
    """
    Returns the edges of the specified friction cone. For 3D vectors, the
    friction cone is approximated by a pyramid whose vertices are circumscribed
    by the friction cone.

    In the case where the friction coefficient is 0, a list containing only the
    original contact force is returned.

    Args:
        f - 2D or 3D contact force.
        mu - friction coefficient.

    Return:
        edges - a list of forces whose convex hull approximates the friction cone.
    """
    # Edge case for frictionless contact
    if mu == 0.:
        return [f]

    # Planar wrenches
    D = f.shape[0]
    if D == 2:
        ########## Your code starts here ##########
        edges = []
        edges = []
        if f[1] != 0 and f[0]==0: 
            edges.append(f + np.array([1, 0])*mu*np.linalg.norm(f))
            edges.append(f + np.array([-1, 0])*mu*np.linalg.norm(f))
        elif f[0] !=0 and f[1]==0:
            edges.append(f + np.array([0, 1, ])*mu*np.linalg.norm(f))
            edges.append(f + np.array([0, -1])*mu*np.linalg.norm(f))
        else:
            if (( cross_matrix(f)[0]*mu) !=0).any(): 
                edges.append(f - cross_matrix(f)[0]*mu)
                edges.append(f + cross_matrix(f)[0]*mu)
            else:
                edges.append(f)

        ########## Your code ends here ##########

    # Spatial wrenches
    elif D == 3:
        ########## Your code starts here ##########
        #edges = [np.zeros(D)] * 4
        edges = []
        if f[2] != 0 and f[0] == 0 and f[1] == 0: 
            edges.append(f + np.array([1, 0, 0])*mu*np.linalg.norm(f))
            edges.append(f + np.array([-1, 0, 0])*mu*np.linalg.norm(f))
            edges.append(f + np.array([0, 1, 0])*mu*np.linalg.norm(f))
            edges.append(f + np.array([0, -1, 0])*mu*np.linalg.norm(f))
        elif f[1] != 0 and f[0] == 0 and f[2] == 0: 
            edges.append(f + np.array([1, 0, 0])*mu*np.linalg.norm(f))
            edges.append(f + np.array([-1, 0, 0])*mu*np.linalg.norm(f))
            edges.append(f + np.array([0, 0, 1])*mu*np.linalg.norm(f))
            edges.append(f + np.array([0, 0, -1])*mu*np.linalg.norm(f))
        elif f[0] !=0 and f[1] == 0 and f[2] == 0: 
            edges.append(f + np.array([0, 1, 0])*mu*np.linalg.norm(f))
            edges.append(f + np.array([0, -1, 0])*mu*np.linalg.norm(f))
            edges.append(f + np.array([0, 0, 1])*mu*np.linalg.norm(f))
            edges.append(f + np.array([0, 0, -1])*mu*np.linalg.norm(f))            
        else:
            n = np.array([f[1], -f[0], 0])/np.linalg.norm(f)
            friction1 = n*mu
            if (friction1 != 0.).any():
                edges.append(f - friction1)
                edges.append(f + friction1)
            else:
                edges.append(f)
            friction2 = cross_matrix(f).dot(n)*mu
            if (friction2 != 0.).all():
                edges.append(f + friction2)
                edges.append(f - friction2)
            else:
                edges.append(f)


        ########## Your code ends here ##########

    else:
        raise RuntimeError("cone_edges(): f must be 3D or 6D. Received a {}D vector.".format(D))

    return edges

def form_closure_program(F):
    """
    Solves a linear program to determine whether the given contact wrenches
    are in form closure.

    Args:
        F - matrix whose columns are 3D or 6D contact wrenches.

    Return:
        True/False - whether the form closure condition is satisfied.
    """
    ########## Your code starts here ##########
    # Hint: you may find np.linalg.matrix_rank(F) helpful
    # TODO: Replace the following program (check the cvxpy documentation)

    r = np.linalg.matrix_rank(F)    
    D = F.shape[0]
    j = F.shape[1]
    if r < D: 
        return False;

    k = cp.Variable(j, pos=True)
    constraints = [F@k==0]
    constraints.append(k>=1)
    objective = cp.Minimize(np.ones(j)@k)

    ########## Your code ends here ##########

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.ECOS)

    return prob.status not in ['infeasible', 'unbounded']

def is_in_form_closure(normals, points):
    """
    Calls form_closure_program() to determine whether the given contact normals
    are in form closure.

    Args:
        normals - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.

    Return:
        True/False - whether the forces are in form closure.
    """
    ########## Your code starts here ##########
    # TODO: Construct the F matrix (not necessarily 6 x 7)
    F = []
    for normal, point in zip(normals, points):
        f = wrench(normal, point)
        F.append(f)
    F= np.stack(F).T
    ########## Your code ends here ##########

    return form_closure_program(F)

def is_in_force_closure(forces, points, friction_coeffs):
    """
    Calls form_closure_program() to determine whether the given contact forces
    are in force closure.

    Args:
        forces - list of 2D or 3D contact forces.
        points - list of 2D or 3D contact points.
        friction_coeffs - list of friction coefficients.

    Return:
        True/False - whether the forces are in force closure.
    """
    ########## Your code starts here ##########
    # TODO: Call cone_edges() to construct the F matrix (not necessarily 6 x 7)
    #F = np.zeros((6,7))    
    print("force:", forces)
    print("points:", points)
    print("friction_coeffs", friction_coeffs)
    W = []
    for force, point, friction_coeff in zip(forces, points, friction_coeffs):
        F = cone_edges(force, friction_coeff)
        for normal in F:
            w = wrench(normal, point)
            W.append(w)

    F = np.stack(W).T

    ########## Your code ends here ##########

    return form_closure_program(F)
