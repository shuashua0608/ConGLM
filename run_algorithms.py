import os
from basic_func import *
from algo.conf import *

pth = "D:/pycharmProject/Con_GLM_compare/data/"
arms = np.load(os.path.join(pth, "synthetic/arm_feats.npy"))
suparms = np.load(os.path.join(pth, "synthetic/suparms.npy"))
users = np.load(os.path.join(pth,"synthetic/users.npy"))



