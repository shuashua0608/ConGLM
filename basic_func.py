import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize, NonlinearConstraint

def solve_MLE(X, y, lam_inv = 1):

    model = LogisticRegression(penalty='l2', C = lam_inv, fit_intercept=False, solver='newton-cg')
    model.fit(X, y)
    return model.coef_.reshape(-1) # theta_t: MLE

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def weighted_norm(x, A):  ## ||x||_A
    return np.sqrt(np.dot(x, np.dot(A, x)))


def get_kappa(arm_norm_ub, param_norm_ub):
    tmp = 1 / (1 + np.exp(-arm_norm_ub * param_norm_ub))
    tmp = tmp * (1 - tmp)
    kappa = 1 / tmp
    return kappa

# fix Xt for each iteration
def pool_index(horizon, pool_size, arms):
    index_list = []
    for i in range(horizon):
        index_pool = np.random.choice(range(0, len(arms)), pool_size, replace=False)
        index_list.append(index_pool)
    return index_list


# burning period #
def init(theta_star, arms, lamb, length, kappa):

    dim = len(theta_star)
    y = []
    X = []
    M = np.eye(dim) * lamb * kappa
    for i in range(length):
        row = np.arange(arms.shape[0])
        np.random.shuffle(row)
        a1 = arms[row[0]]
        a2 = arms[row[1]]
        d_i = a1 - a2
        r_i = sigmoid(np.dot(d_i, theta_star))
        p = np.array([1 - r_i, r_i])
        X.append(d_i.reshape(dim))
        y.append(int(np.random.choice([0, 1], p=p.ravel())))
        M = M + np.outer(d_i, d_i)

    return X, y, M

"""
projection: if theta_t(MLE) falls out of parameter space, 
we need to project theta_t^{1} = argmin\|gt(theta) - gt(theta_t)\|_M_t^{-1}
"""

def gt(theta, arms, lamb):
    g = sigmoid(arms @ theta)
    g = np.dot(g, arms) + lamb * theta
    return g


def proj_fun(theta, theta_t, arms, M, lamb):
    diff_gt = gt(theta, arms, lamb) - gt(theta_t, arms, lamb)
    fun = weighted_norm(diff_gt, np.linalg.inv(M))
    return fun


def hessian(theta, arms, lamb):
    dim = len(theta)
    g = sigmoid(arms @ theta)
    coeffs = np.multiply(g, 1 - g)
    res = np.dot(np.matrix(arms).T, np.dot(np.diag(coeffs), np.matrix(arms))) + lamb * np.eye(dim)
    return res


def proj_grad(theta, theta_t, arms, lamb, M):
    diff_gt = gt(theta, arms, lamb) - gt(theta_t, arms, lamb)
    grads = 2 * np.dot(np.linalg.inv(M), np.dot(hessian(theta, arms, lamb), diff_gt).T)
    return grads


def projection(arms, theta_t, M, lamb, param_norm_ub, dim):
    fun = lambda t: proj_fun(t, theta_t, arms, M, lamb)
    grads = lambda t: proj_grad(t, theta_t, arms, lamb, M)
    norm = lambda t: np.linalg.norm(t)
    constraint = NonlinearConstraint(norm, 0, param_norm_ub)
    opt = minimize(fun, x0=np.zeros(dim), constraints=constraint)
    return opt.x

