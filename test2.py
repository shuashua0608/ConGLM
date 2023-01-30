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


regret_maxinp_all = []
regret_conglm_all_n0 = []
regret_conglm_all_5 = []
regret_conglm_all_3 = []
regret_conglm_all_1 = []
regret_conglm_all_lin5 = []
regret_conglm_all_lin3 = []
regret_conglm_all_lin1 = []



for theta in users:

    info_maxinp = MaxInp(arms, maxinp_para, theta, pool_index_list, horizon)

    info_conglm_no = ConGLM(arms, suparms, conglm_para, theta, lambda t:0, pool_index_list, horizon)

    info_conglm_log5 = ConGLM(arms, suparms, conglm_para, theta, bt_log_5, pool_index_list, horizon)

    info_conglm_log3 = ConGLM(arms, suparms, conglm_para, theta, bt_log_3, pool_index_list, horizon)

    info_conglm_log1 = ConGLM(arms, suparms, conglm_para, theta, bt_log_1, pool_index_list, horizon)

    info_conglm_lin5 = ConGLM(arms, suparms, conglm_para, theta, bt_lin_5, pool_index_list, horizon)

    info_conglm_lin3 = ConGLM(arms, suparms, conglm_para, theta, bt_lin_3,pool_index_list, horizon)

    info_conglm_lin1 = ConGLM(arms, suparms, conglm_para, theta, bt_lin_1, pool_index_list, horizon)


    regret_maxinp = info_maxinp["regret"]
    regret_conglm_no = info_conglm_no["regret"]
    regret_conglm_5 = info_conglm_log5["regret"]
    regret_conglm_3 = info_conglm_log3["regret"]
    regret_conglm_1 = info_conglm_log1["regret"]
    regret_conglm_lin5 = info_conglm_lin5["regret"]
    regret_conglm_lin3 = info_conglm_lin3["regret"]
    regret_conglm_lin1 = info_conglm_lin1["regret"]

    regret_maxinp_all.append(max(regret_maxinp))
    regret_conglm_all_n0.append(max(regret_conglm_no))
    regret_conglm_all_5.append(max(regret_conglm_5))
    regret_conglm_all_3.append(max(regret_conglm_3))
    regret_conglm_all_1.append(max(regret_conglm_1))
    regret_conglm_all_lin5.append(max(regret_conglm_lin5))
    regret_conglm_all_lin3.append(max(regret_conglm_lin5))
    regret_conglm_all_lin1.append(max(regret_conglm_lin1))

print("mean maxinp:", np.average(regret_maxinp_all))
print("mean conglm_no:", np.average(regret_conglm_all_n0))
print("mean conglm_log5:", np.average(regret_conglm_all_5))
print("mean conglm_log3:", np.average(regret_conglm_all_3))
print("mean conglm_log1:", np.average(regret_conglm_all_3))

print("mean conglm_lin5:", np.average(regret_conglm_all_lin5))
print("mean conglm_lin3:", np.average(regret_conglm_all_lin3))
print("mean conglm_lin1:", np.average(regret_conglm_all_lin1))

pth = "/data/ysh/tmp"
output = os.path.join(pth, "regret_all_users")
if not os.path.exists(output):
    os.mkdir(output)

np.save(os.path.join(output, "regret_maxinp_all"), regret_maxinp_all)
np.save(os.path.join(output, "regret_conglm_all_n0"), regret_conglm_all_n0)
np.save(os.path.join(output, "regret_conglm_all_5"), regret_conglm_all_5)
np.save(os.path.join(output, "regret_conglm_all_3"), regret_conglm_all_3)
np.save(os.path.join(output, "regret_conglm_all_1"), regret_conglm_all_1)
np.save(os.path.join(output, "regret_conglm_lin3"), regret_conglm_all_lin3)
np.save(os.path.join(output, "regret_conglm_lin5"), regret_conglm_all_lin5)
np.save(os.path.join(output, "regret_conglm_lin1"), regret_conglm_all_lin1)