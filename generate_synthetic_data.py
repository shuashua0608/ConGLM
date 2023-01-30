import collections
import os
import random

import numpy as np
import numpy.linalg as LA


def generate_dataset(dim=5, num_suparms=20, num_arms=100, num_users=100, M = 4):

    random.seed(1234)

    relation = collections.defaultdict(list)

    ### item feature matrix ####
    X = np.ones([num_arms, dim])
    user_M = np.ones([num_users, dim])

    for i in range(num_arms):
        X[i] = np.random.uniform(-1/np.sqrt(dim), 1/np.sqrt(dim), dim)
        X[i]/=LA.norm(X[i])
    for user_id in range(num_users):
        user_M[user_id] = np.random.uniform(-1/np.sqrt(dim), 1/np.sqrt(dim), dim)
        

    #### arm feature matrix ####
    suparm_X = np.ones([num_suparms, dim])

    for i in range(num_arms):

 
        num_related_suparms = random.randint(1, M)
        related_suparms = random.sample(range(num_suparms), num_related_suparms)

        for suparm_id in related_suparms:
            relation[i].append(suparm_id)

    for i in range(num_suparms):
        ni = 0
        fea_i = np.zeros(dim)
        for j in range(num_arms):
            if (i in relation[j]):
                ni = ni + 1 / len(relation[j])
                fea_i = fea_i + X[j] / len(relation[j])
        suparm_X[i] = fea_i / ni

        #### generate user parameter matrix   

    return [X, suparm_X, relation, user_M]


X, suparm_X, relation, user_M = generate_dataset(dim=5, num_suparms=20, num_arms=100, num_users=100, M=4)







