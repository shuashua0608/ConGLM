import os
import numpy as np
from algo.ConGLM import ConGLM
from algo.MaxInp import MaxInp
from algo.conf import conglm_para
from algo.conf import maxinp_para
import matplotlib.pyplot as plt
from basic_func import *
import math
###############################################################################################
pth = "D:/pycharmProject/Con_GLM_compare/data/"
arms = np.load(os.path.join(pth, "synthetic/arm_feats.npy"))
suparms = np.load(os.path.join(pth, "synthetic/suparms.npy"))
users = np.load(os.path.join(pth,"synthetic/users.npy"))

horizon = 2000
pool_size = 20
pool_index_list = pool_index(horizon, pool_size, arms)
# conversation function #
bt_log_5 = lambda t:5 * int(math.log(t + 1))
bt_log_3 = lambda t:3 * int(math.log(t + 1))
bt_log_1 = lambda t: int(math.log(t + 1))
suparms_time = lambda t: 5* np.log(t+1)
bt_lin_5 = lambda t: int(0.5*t)
bt_lin_3 = lambda t: int(0.3*t)
bt_lin_1 = lambda t: int(0.1*t)

theta_star = users[1]
info_maxinp = MaxInp(arms, maxinp_para, theta_star, pool_index_list, horizon)
info_conglm = ConGLM(arms, suparms, conglm_para, theta_star, pool_index_list, bt_lin_1, horizon)

regret_maxinp = info_maxinp["regret"]
regret_conglm = info_conglm["regret"]

plt.plot(regret_maxinp)
plt.plot(regret_conglm)
plt.show()



