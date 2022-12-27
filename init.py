import numpy as np

from logistic_bandit_algo import LogisticBandit
from numpy.linalg import solve, slogdet
from scipy.optimize import minimize, NonlinearConstraint
from logistic_env import sigmoid, dsigmoid, weighted_norm,get_para_constant
from sklearn.linear_model import LogisticRegression
from generate_data import generate_dataset

class ConGLM(LogisticBandit):

    ## Hoeffding Inequality ##
    def __init__(self, param_norm_ub, arm_norm_ub, kappa, lamb, dim, failure_level,do_proj=False):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)

        self.name = 'Con-GLM'
        self.do_proj = do_proj
        # initialize some learning attributes
        self.lamb = lamb
        self.design_matrix = self.lamb * np.eye(self.dim)
        self.design_matrix_inv = (1 / self.lamb) * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.theta_tilde = np.random.normal(0, 1, (self.dim,))
        self.ctr = 0
        self.ucb_bonus = 0
        self.kappa = kappa
        # containers
        self.arms = []
        self.rewards = []

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.design_matrix = self.lamb * np.eye(self.dim)
        self.design_matrix_inv = (1 / self.lamb) * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.theta_tilde = np.random.normal(0, 1, (self.dim,))
        self.ctr = 0
        self.arms = []
        self.rewards = []

    def learn(self, arm, reward):
        """
        Update the MLE, project if required/needed.
        """
        self.arms.append(arm)
        self.rewards.append(reward)

        # learn the m.l.e by iterative approach (a few steps of Newton descent)

        theta_hat = self.theta_hat
        model = LogisticRegression(penalty='l2', C = 1/self.lamb, fit_intercept = False, solver = 'newton-cg')
        model.fit(self.arms, self.rewards)
        self.theta_hat = model.coef_.reshape(-1)

        # update counter
        self.ctr += 1

        # perform projection (if required)
        if self.do_proj and len(self.rewards) > 2:
            if np.linalg.norm(self.theta_hat) < self.param_norm_ub:
                self.theta_tilde = self.theta_hat
            else:
                self.theta_tilde = self.projection(self.arms)
        else:
            self.theta_tilde = self.theta_hat

    def pull(self, arm_set):
        # update bonus bonus
        self.update_ucb_bonus()

        # find optimistic arm
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        # update design matrix and inverse
        self.design_matrix += np.outer(arm, arm)
        self.design_matrix_inv += -np.dot(self.design_matrix_inv, np.dot(np.outer(arm, arm), self.design_matrix_inv)) \
                                  / (1 + np.dot(arm, np.dot(self.design_matrix_inv, arm)))
        return arm


    def pull(self, arm_set, suparm_set):

        # Basic function for pulling arm

        self.update_ucb_bonus()

        # find optimistic arm
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        # update design matrix and inverse
        self.design_matrix += np.outer(arm, arm)
        self.design_matrix_inv += -np.dot(self.design_matrix_inv, np.dot(np.outer(arm, arm), self.design_matrix_inv)) \
                                  / (1 + np.dot(arm, np.dot(self.design_matrix_inv, arm)))


    def choose_suparm(self, suparm, arm_set):
        """
        choose optimal suparm
        """
        result_a = []
        for a in arm_set:
            result_a = np.dot(np.dot(a, self.design_matrix_inv), suparm)
            norm_M = np.linalg.norm(result_a)^2
            result_a.append(norm_M)

        result_b = 1 + np.dot(np.dot(suparm.T, self.design_matrix_inv), suparm)

        return sum(result_a)/(len(result_a)* result_b)


    def update_ucb_bonus(self):
        """
        Updates the UCB bonus.
        """
        logdet = slogdet(self.design_matrix)[1]
        res = np.sqrt(2 * np.log(1 / self.failure_level) + logdet - self.dim * np.log(self.lamb/self.kappa))
        res *= 0.5 / self.kappa
        res += 0.5 * np.sqrt(self.lamb/self.kappa)*self.param_norm_ub
        self.ucb_bonus = res

    def compute_optimistic_reward(self, arm):
        """
        Computes the UCB.
        """
        norm = weighted_norm(arm, self.design_matrix_inv)
        pred_reward = sigmoid(np.sum(self.theta_tilde * arm))
        bonus = self.ucb_bonus * norm
        return pred_reward + bonus

    def proj_fun(self, theta, arms):
        """
        Filippi et al. projection function
        """
        diff_gt = self.gt(theta, arms) - self.gt(self.theta_hat, arms)
        fun = np.dot(diff_gt, np.dot(self.design_matrix_inv, diff_gt))
        return fun

    def proj_grad(self, theta, arms):
        """
        Filippi et al. projection function gradient
        """
        diff_gt = self.gt(theta, arms) - self.gt(self.theta_hat, arms)
        grads = 2 * np.dot(self.design_matrix_inv, np.dot(self.hessian(theta, arms), diff_gt))
        return grads

    def gt(self, theta, arms):
        coeffs = sigmoid(np.dot(arms, theta))[:, None]
        res = np.sum(arms, coeffs) + self.lamb * self.kappa * theta
        return res

    def hessian(self, theta, arms):
        coeffs = dsigmoid(np.dot(arms, theta))[:, None]
        res = np.dot(np.array(arms).T, coeffs * arms) + self.lamb * self.kappa * np.eye(self.dim)
        return res

    def projection(self, arms):
        fun = lambda t: self.proj_fun(t, arms)
        grads = lambda t: self.proj_grad(t, arms)
        norm = lambda t: np.linalg.norm(t)
        constraint = NonlinearConstraint(norm, 0, self.param_norm_ub)
        opt = minimize(fun, x0=np.zeros(self.dim), method='SLSQP', jac=grads, constraints = constraint)
        return opt.x



if __name__=='__main__':

    ## generate data ##
    X, suparm_X, relation, user_M = generate_dataset(dim=5, num_arms = 150,num_suparms = 50, num_users=100, M = 4)

    # generate kappa constant #

    kappa, arm_norm_ub, param_norm_ub = get_para_constant(X, suparm_X, user_M)

    ###########################################
    dim = len(user_M[0,:])

    LS = ConGLM(param_norm_ub, arm_norm_ub, kappa, dim, lamb = 10, failure_level = 0.025, do_proj = True)

    result = LS.pull(X)

    print(result)
