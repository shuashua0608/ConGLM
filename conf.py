import math
import numpy as np

maxinp_para = {"lamb": 1, "sigma":0.05,"arm_norm_ub": 1,"param_norm_ub":2,
               "eta": 1,"horizon": 2000, "init": 20}

conglm_para = {"lamb": 10, "sigma": 0.05,"arm_norm_ub": 1, "param_norm_ub":2,
               "suparm_strategy": "barycentric spanner","alpha":0.5,
               "horizon": 2000, "init": 20}

