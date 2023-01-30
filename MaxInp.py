from basic_func import *
import random
import numpy.linalg as LA
"""
eta is also treated as hyperparameter

def get_constant_eta(dim, kappa, sigma, t):
    eta = np.sqrt(0.5 * dim * np.log(1 + 2 * t / dim) + np.log(1 / sigma))
    eta *= 0.5 * kappa
    return eta
"""

def choose_max_pair(X_pool, theta, eta, M):
    C_t = []
    for arm in X_pool:
        if [np.dot(arm - arm_other, theta)
            + eta * weighted_norm(arm - arm_other, LA.inv(M)) > 0 for arm_other in X_pool]:
            C_t.append(arm)

    arm_1 = C_t[random.randint(0, len(C_t) - 1)]
    uncertainty = [weighted_norm(arm - arm_1, np.linalg.inv(M)) for arm in C_t]
    arm_2 = C_t[np.argmax(uncertainty)]
    diff_arm = arm_1 - arm_2
    return arm_1, arm_2, diff_arm


def MaxInp(arm, conf, theta_star, pool_index_list, horizon):
    # conf = conglm_para

    param_norm_ub = conf["param_norm_ub"]
    arm_norm_ub = conf["arm_norm_ub"]
    lamb = conf["lamb"]
    eta = conf["eta"]
    init_length = conf["init"]

    kappa = get_kappa(arm_norm_ub, param_norm_ub)
    dim = len(theta_star)
    # burning period #
    x, y, M = init(theta_star, arm, lamb, init_length, kappa)

    regret_s = []  # strong regret
    regret = []  # mean regret
    reward = []
    theta_record = []

    for i in range(horizon):
        # conversation part #

        theta = solve_MLE(x, y, 1 / lamb)  # theta: MLE estimate
        theta_record.append(theta)

        X_pool = arm[pool_index_list[i]]
        mus = sigmoid(X_pool @ theta_star)
        mu_1 = np.max(mus)

        a1, a2, d_i = choose_max_pair(X_pool, theta, eta, M)
        M = M + np.outer(d_i, d_i)
        r_i = sigmoid(np.dot(d_i, theta_star))
        p = np.array([1 - r_i, r_i])
        x.append(d_i.reshape(dim))
        y.append(int(np.random.choice([0, 1], p=p.ravel())))

        reward_1 = sigmoid(np.dot(a1, theta_star))
        reward_2 = sigmoid(np.dot(a2, theta_star))
        reward.append(0.5 * (reward_1 + reward_2))

        regret_s.append(max(mu_1 - reward_1, mu_1 - reward_2))
        regret.append(mu_1 - 0.5 * (reward_1 + reward_2))

    cum_reward = [sum(reward[0:i]) for i in range(len(reward))]
    cum_regret_s = [sum(regret_s[0:i]) for i in range(len(regret_s))]
    #cum_regret_w = [sum(regret_w[0:i]) for i in range(len(regret_w))]
    cum_regret = [sum(regret[0:i]) for i in range(len(regret))]

    info_all = {"reward": cum_reward, "regret": cum_regret, "regret_strong": cum_regret_s, "theta": theta_record}

    return info_all