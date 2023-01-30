import numpy as np
import math
from sklearn.linear_model import LogisticRegression
import numpy.linalg as LA
from basic_func import *
import random

# TODO: a more accarate method to estimate the MLE parameter #
# TODO: how to compute barycentric spanner set: B #
# TODO: Test if conversation part (suparms) can improve our method #

"""
Barycentric spanner:
assume key-term infomation is contained in a subset barycentric spanner B;
each time select a pair of key-terms independently from B.
"""

"""
A simple way to construct barycentric spanner: choose best dim/2 suparms and worst dim/2 suparms
(performance is not good)
"""
def compute_B(suparms, theta_star):
# a simple way to construct barycentric spanner for suparms #
# barycentric spanner B only contains n = dim suparms, assume all infomation is contained in B #
    info_k = []
    dim = len(theta_star)
    for k in suparms:
        tmp = sigmoid(np.dot(k, theta_star))
        info_k.append(tmp)
    ind1 = np.array(info_k).argsort()[-dim:]
    ind2 = np.array(info_k).argsort()[0:dim]
    index = np.concatenate((ind1, ind2))
    return suparms[index]


def choose_suparm_pair(suparms, suparm_strategy, M, B):

    suparm_1 = suparm_2 = diff_suparm = None
    if suparm_strategy == "random":
        row = np.arange(suparms.shape[0])
        np.random.shuffle(row)
        ind = row[0:2]
        suparm_1 = suparms[ind[0]]
        suparm_2 = suparms[ind[1]]
        diff_suparm = suparm_1 - suparm_2

    if suparm_strategy == "barycentric spanner":
        row = np.arange(B.shape[0])
        np.random.shuffle(row)
        ind = row[0:2]
        suparm_1 = B[ind[0]]
        suparm_2 = B[ind[1]]
        diff_suparm = suparm_1 - suparm_2

    if suparm_strategy == "max info":
        suparm_norm = [weighted_norm(x, LA.inv(M)) for x in suparms]
        suparm_1 = suparms[np.argmax(suparm_norm)]
        diff_norm = [weighted_norm(x - suparm_1, LA.inv(M)) for x in suparms]
        suparm_2 = suparms[np.argmax(diff_norm)]
        diff_suparm = suparm_1 - suparm_2

    return suparm_1, suparm_2, diff_suparm


"""
! alpha is viewed as hyperparameter and not updated

def get_constant_alpha(theta_star, M, lamb, dim, delta, kappa):
    # calculate alpha_t for each round: alpha_t is parameter for the confidence bound
    S = LA.norm(theta_star)
    det_a = np.sqrt(LA.det(M))
    alpha = np.sqrt(2 * np.log(det_a / (delta * math.pow(lamb * kappa, dim / 2))))
    alpha += np.sqrt(lamb * kappa) * S
    alpha = kappa * alpha
    return alpha
"""

# select a pair of arms from X_pool
def choose_arm_pair(X_pool, theta, alpha, M):

    C_t = []
    for arm in X_pool:
        if [(np.dot(arm - arm_other, theta))+
            alpha * weighted_norm(arm - arm_other, LA.inv(M)) > 0 for arm_other in X_pool]:
            C_t.append(arm)

    arm_1 = C_t[random.randint(0, len(C_t) - 1)]
    uncertainty = [weighted_norm(arm - arm_1, LA.inv(M)) for arm in C_t]
    arm_2 = C_t[np.argmax(uncertainty)]
    diff_arm = arm_1 - arm_2
    return arm_1, arm_2, diff_arm


def ConGLM(arms, suparms, conf, theta_star, pool_index_list, buget_func, horizon):
    # conf = conglm_para
    suparm_strategy = conf["suparm_strategy"]
    param_norm_ub = conf["param_norm_ub"]
    arm_norm_ub = conf["arm_norm_ub"]
    lamb = conf["lamb"]
    alpha = conf["alpha"]
    init_length = conf["init"]

    kappa = get_kappa(arm_norm_ub, param_norm_ub)
    dim = len(theta_star)
    # burning period #
    x, y, M = init(theta_star, arms, lamb, init_length, kappa)
    B = compute_B(suparms, theta_star)
    regret_s = []  # strong regret
    # regret_w = [] # weak regret
    regret = []  # mean regret
    reward = []
    theta_record = []

    for i in range(horizon):
        # conversation part #
        if buget_func(i + 1) - buget_func(i) > 0:
            conv_times = int(buget_func(i + 1) - buget_func(i))
            for j in range(conv_times):
                suparm_1, suparm_2, d_i = choose_suparm_pair(suparms, suparm_strategy, M, B)
                M = M + np.outer(d_i, d_i)
                r_i = sigmoid(np.dot(d_i, theta_star))
                p = np.array([1 - r_i, r_i])
                x.append(d_i.reshape(dim))
                y.append(int(np.random.choice([0, 1], p=p.ravel())))

        theta = solve_MLE(x, y, 1 / lamb)  # theta: MLE estimate
        theta_record.append(theta)
        """
        if LA.norm(theta)> param_norm_ub:
            theta = projection(arm, theta, M, lamb, param_norm_ub, dim)
        """
        X_pool = arms[pool_index_list[i]]
        mus = sigmoid(X_pool @ theta_star)
        mu_1 = np.max(mus)

        # alpha = get_constant_alpha(theta_star, M, lamb, dim, sigma, kappa)
        a1, a2, d_i = choose_arm_pair(X_pool, theta, alpha, M)
        M = M + np.outer(d_i, d_i)
        r_i = sigmoid(np.dot(d_i, theta_star))
        p = np.array([1 - r_i, r_i])
        x.append(d_i.reshape(dim))
        y.append(int(np.random.choice([0, 1], p=p.ravel())))

        reward_1 = sigmoid(np.dot(a1, theta_star))
        reward_2 = sigmoid(np.dot(a2, theta_star))
        reward.append(0.5 * (reward_1 + reward_2))
        regret_s.append(max(mu_1 - reward_1, mu_1 - reward_2))
        # regret_w.append(min(mu_1 - reward_1, mu_1 - reward_2))
        regret.append(mu_1 - 0.5 * (reward_1 + reward_2))

    cum_reward = [sum(reward[0:i]) for i in range(len(reward))]
    cum_regret_s = [sum(regret_s[0:i]) for i in range(len(regret_s))]
    # cum_regret_w = [sum(regret_w[0:i]) for i in range(len(regret_w))]
    cum_regret = [sum(regret[0:i]) for i in range(len(regret))]

    info_all = {"reward": cum_reward, "regret": cum_regret, "regret_strong": cum_regret_s, "theta": theta_record}

    return info_all