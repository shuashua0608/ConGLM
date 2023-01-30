from sklearn.linear_model import LogisticRegression
import math
import numpy as np


#TODO: How to get a more accurate estimation for MLE theta
# Current method: sklearn.linear_model.LogisticRegression #
#TODO: sgd method is faster?

def solve_MLE(X, y, lam_inv=1):
    model = LogisticRegression(penalty='l2', C=lam_inv, fit_intercept=False, solver='newton-cg')
    model.fit(X, y)
    return model.coef_.reshape(-1)  # theta_t: MLE

#TODO: check if conversation can improve ConGLM
# Compare different conversation frequency #
# conversation function #

bt_none = lambda t:0
bt_log_20 = lambda t:20 * int(math.log(t + 1))
bt_log_10 = lambda t:10 * int(math.log(t + 1))
bt_log_5 = lambda t: 5 * int(math.log(t + 1))

bt_lin_5 = lambda t: int(0.5*t)
bt_lin_3 = lambda t: int(0.3*t)
bt_lin_1 = lambda t: int(0.1*t)

#TODO: Dueling bandit: each time choose a pair of arms/suparms to get a relative feedback
# For arms, the first arm is selected from set C containing "good" arms;
# the second arm is selected with maximum uncertainty.
# For suparms, select a pair of suparms from barycentric spanner;
# or select a pair of suparms follows maximum information.#

#TODO: real dataset description:
# 2000 arms, 400 users, 50 dimension
# Affinity: user preference for arms (mu_x = x^T\theta_*),
# arm_to_suparm: arm & key-term relationship
# 2 steps for preprocessing dataset:
# (1). cluster arms into less than 100 groups (dueling bandit can't handle large dataset)
# (2). use PCA to reduce dimension from 50 dim to 10 dim for the cluster centroids
# (3). recalculate key-terms and user preference using affinity and arm_to_suparm.