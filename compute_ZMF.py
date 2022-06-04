import os

from operator import itemgetter
from numpy import random, zeros
import numpy as np
from scipy.spatial.distance import cosine
from scipy.linalg import norm
from sklearn.linear_model import LinearRegression, Lasso
from math import log
import matplotlib.pyplot as plt
import pandas as pd


def compute_user_rank(user_item_ratings):

    user_rank = {}
    for user_id in user_item_ratings.keys():
        for item_id in user_item_ratings[user_id]:
            user_rank[user_id] = user_rank.get(user_id, 0.0) + 1.0

    user_rank_list = []
    for user_id in user_rank.keys():
        user_rank_list.append((user_id, user_rank[user_id]))

    user_rank_list = sorted(user_rank_list, key=itemgetter(1))

    user_rank_dict = {}
    for user_pair in user_rank_list:
        user_rank_dict[user_pair[0]] = user_pair[1]

    return user_rank_dict

def compute_item_rank(user_item_ratings):

    item_rank = {}
    for user_id in user_item_ratings.keys():
        for item_id in user_item_ratings[user_id]:
            item_rank[item_id] = item_rank.get(item_id, 0.0) + 1.0

    item_rank_list = []
    for item_id in item_rank.keys():
        item_rank_list.append((item_id, item_rank[item_id]))

    item_rank_list = sorted(item_rank_list, key=itemgetter(1))

    item_rank_dict = {}
    for item_pair in item_rank_list:
        item_rank_dict[item_pair[0]] = item_pair[1]

    return item_rank_dict

def optimize_function_err_fun(train_data_dict, test_data_dict, user_rank_dict, item_rank_dict, iter_no, eta, beta):

    user_features = {}
    item_features = {}

    (user_coef, rank_list) = compute_alpha(train_data_dict, test_data_dict, item_rank_dict)

    for x in range(0, iter_no):

        print('Iteration # %s' % x)

        user_id_list = random.random_sample(100) * train_data_dict.keys().__len__()

        user_len = max(train_data_dict.keys()) + 1
        item_len = max(item_rank_dict.keys()) + 1

        u = random.random([user_len, 30])
        v = random.random([item_len, 30])

        user_repo = list(train_data_dict.keys())

        for uid_id in user_id_list:

            user_id = user_repo[int(uid_id)]
            
            if train_data_dict[user_id].__len__() < 1:
                continue

            item_id_list = list(train_data_dict[user_id].keys())
            sampled_item_id_list = random.random_sample(10) * item_id_list.__len__()

            item_repo = train_data_dict[user_id].keys()

            item_counter = 0

            for iid in sampled_item_id_list:

                #item_counter += 1
                #print('Item Counter %s' % item_counter)

                item_id = item_id_list[int(iid)]

                R_v = train_data_dict[user_id][item_id_list[int(iid)]]*1.0/5
                
                t_0 = np.dot(u[user_id], v[item_id])
                t_1 = norm(u[user_id])
                t_2 = norm(v[item_id])
                t_3 = t_1 * t_2
                t_4 = t_0

                rank = user_coef[user_id] * t_0
                if rank < 1.0:
                    rank = 1.0
               
                u[user_id] -= -1.0 * eta *(2*(R_v - t_0/t_3))/t_3 * v[item_id]
                u[user_id] -= eta*2.0*t_4*(R_v - t_4/t_3)/(t_1**3*t_2) * u[user_id]

                t_0 = np.dot(v[item_id], u[user_id])
                t_1 = norm(u[user_id])
                t_2 = norm(v[item_id])
                t_3 = t_1 * t_2
                t_4 = 2*(R_v-t_0/t_3) 
                v[item_id] -= eta*beta*item_len/(log(rank/item_len)**2*t_0)*u[user_id] - eta*(t_4/t_3*u[user_id] - (t_0*t_4)/(t_1*t_2**3)*v[item_id])

                user_features[user_id] = u[user_id]
                item_features[item_id] = v[item_id]

    return user_features, item_features

def compute_mf(train_data_dict, test_data_dict, item_rank_dict, eta):

    user_len = max(train_data_dict.keys()) + 1
    item_len = max(item_rank_dict.keys()) + 1

    u = random.random([user_len, 30])
    v = random.random([item_len, 30])

    user_id_list = random.random_sample(100) * train_data_dict.keys().__len__()
    user_repo = list(train_data_dict.keys())

    user_features = {}
    item_features = {}

    for uid_id in user_id_list:

        user_id = user_repo[int(uid_id)]
            
        if train_data_dict[user_id].__len__() < 1:
            continue

        item_id_list = list(train_data_dict[user_id].keys())
        sampled_item_id_list = random.random_sample(10) * item_id_list.__len__()

        item_repo = train_data_dict[user_id].keys()

        for iid in sampled_item_id_list:

            item_id = item_id_list[int(iid)]

            R_v = train_data_dict[user_id][item_id_list[int(iid)]]
    
            u[user_id] += eta*2*(R_v - np.dot(u[user_id], v[item_id])) * v[item_id]
            v[item_id] += eta*2*(R_v - np.dot(u[user_id], v[item_id])) * u[user_id]

            user_features[user_id] = u[user_id]
            item_features[item_id] = v[item_id]

    pr_dict = {}
    pr_list = []

    mae = 0.0
    total_no = 0.0

    for user_id in test_data_dict.keys():
        for item_id in test_data_dict[user_id]:
            if user_id in user_features and item_id in item_features:
                R_v = 5.0 * (np.dot(u[user_id], v[item_id])/(norm(u[user_id])*norm(v[item_id])))
                pr_dict[item_id] = pr_dict.get(item_id, 0)+1
                mae += abs(R_v - test_data_dict[user_id][item_id])
                total_no += 1

    for item_id in pr_dict.keys():
        pr_list.append((item_id, pr_dict[item_id]))

    pr_list_s = sorted(pr_list, key=itemgetter(1), reverse=True)
    rank_list = []

    iter_id = 0
    rank_id = 1
    while iter_id < pr_list_s.__len__():
        rank_list.append(rank_id)
        while iter_id+1 < pr_list_s.__len__() and pr_list_s[iter_id] == pr_list_s[iter_id+1]:
            rank_list.append(rank_id)
            iter_id += 1
        rank_id += 1
        iter_id += 1

    DMF = 0.0
    for rank_val in rank_list:
        DMF += log(rank_val*1.0/rank_list[-1])
    DMF = 1 + rank_list.__len__()/DMF

    print('DMF:%s' % DMF)

    return (mae/total_no, DMF)

def compute_alpha(train_data_dict, test_data_dict, item_rank_dict):

    eta = 1e-4

    user_len = max(train_data_dict.keys()) + 1
    item_idx_list = list(item_rank_dict.keys())
    item_len = item_idx_list.__len__()

    u = random.random([user_len, 30])
    v = random.random([item_len, 30])

    user_id_list = random.random_sample(100) * train_data_dict.keys().__len__()
    user_repo = list(train_data_dict.keys())

    for uid_id in user_id_list:

        user_id = user_repo[int(uid_id)]
            
        if train_data_dict[user_id].__len__() < 1:
            continue

        item_id_list = list(train_data_dict[user_id].keys())
        sampled_item_id_list = random.random_sample(10) * item_id_list.__len__()

        item_repo = train_data_dict[user_id].keys()

        for iid in sampled_item_id_list:

            item_id = item_id_list[int(iid)]

            R_v = train_data_dict[user_id][item_id_list[int(iid)]]

            u[user_id] += eta*2*(R_v - np.dot(u[user_id], v[int(iid)])) * v[int(iid)]
            v[int(iid)] += eta*2*(R_v - np.dot(u[user_id], v[int(iid)])) * u[user_id]

    X = zeros([item_len, user_len]) 
    Y = zeros([item_len, 1])
    for id_idx in range(0, item_len):
        Y[id_idx] = item_idx_list[id_idx]
        for user_id in range(0, user_len):
            X[id_idx][user_id] = np.dot(u[user_id], v[id_idx])

    print('Computing Lasso ...')
    LR = Lasso(alpha=1e-7).fit(X, Y)

    user_coef = LR.coef_

    with open('USER_COEF.txt', 'w') as FILE:
        for coef_val in user_coef:
            FILE.write(str(coef_val)+'\n')

    rank_list = {}
    for item_id in range(0, item_len):
        rank_list[item_idx_list[item_id]] = np.dot(X[item_id][:], user_coef) 

    print('Completing compute_alpha ...')

    return (user_coef, rank_list)

def predict_mf(test_data_dict, total_item_list, u, v):

    item_dict = {}
    pr_dict = {}
    pr_list = []

    mae = 0.0
    total_no = 0.0

    for user_id in test_data_dict.keys():
        for item_id in test_data_dict[user_id]:
            if user_id in user_features and item_id in item_features:
                R_v = 5.0 * (np.dot(u[user_id], v[item_id])/(norm(u[user_id])*norm(v[item_id])))
                pr_dict[item_id] = pr_dict.get(item_id, 0)+1
                mae += abs(R_v - test_data_dict[user_id][item_id])
                total_no += 1

    for item_id in pr_dict.keys():
        pr_list.append((item_id, pr_dict[item_id]))

    pr_list_s = sorted(pr_list, key=itemgetter(1), reverse=True)
    rank_list = []

    iter_id = 0
    rank_id = 1
    while iter_id < pr_list_s.__len__():
        rank_list.append(rank_id)
        while iter_id+1 < pr_list_s.__len__() and pr_list_s[iter_id] == pr_list_s[iter_id+1]:
            rank_list.append(rank_id)
            iter_id += 1
        rank_id += 1
        iter_id += 1

    DME = 0.0
    for rank_val in rank_list:
        DME += log(rank_val*1.0/rank_list[-1])
    DME = 1 + rank_list.__len__()/DME

    print('DME: %s' % DME)

    return (mae*1.0/total_no, DME)

if __name__ == '__main__':

    input_file = 'ml-latest-small/ratings.csv'

    user_item_ratings = {}

    with open(input_file, 'r') as FILE:
        for line in FILE:
            data_rec = line.strip().split(',')
            user_id = int(data_rec[0])
            item_id = int(data_rec[1])
            if user_id not in user_item_ratings:
                user_item_ratings[user_id] = {}
            user_item_ratings[user_id][item_id] = float(data_rec[2])

    train_set = {}
    test_set = {}

    train_set_list = []
    test_set_list = []

    train_set_dict = {}
    test_set_dict = {}

    for user_id in user_item_ratings.keys():
        item_list = [item_id for item_id in user_item_ratings[user_id].keys()]
        train_set.setdefault(user_id, [])
        train_set_dict.setdefault(user_id, {})
        test_set_dict.setdefault(user_id, {})

        if item_list.__len__() > 8:
            train_set[user_id] = item_list[:-4]
            for item_id in item_list[:-4]:
                train_set_list.append((user_id, item_id, user_item_ratings[user_id][item_id]))
                train_set_dict[user_id][item_id] = user_item_ratings[user_id][item_id]
            for x in range(-4, 0):
                test_set_list.append((user_id, item_list[x], user_item_ratings[user_id][item_list[x]]))
                test_set_dict[user_id][item_list[x]] = user_item_ratings[user_id][item_list[x]]

        if item_list.__len__() > 4:
            train_set[user_id] = item_list[:-2]
            for item_id in item_list[:-2]:
                train_set_list.append((user_id, item_id, user_item_ratings[user_id][item_id]))
                train_set_dict[user_id][item_id] = user_item_ratings[user_id][item_id]
            for x in range(-2, 0):
                test_set_list.append((user_id, item_list[x], user_item_ratings[user_id][item_list[x]]))
                test_set_dict[user_id][item_list[x]] = user_item_ratings[user_id][item_list[x]]

    with open('train_set.txt', 'w') as FILE:
        for data_rec in train_set_list:
            FILE.write('%s\t%s\t%s\n'%(data_rec[0], data_rec[1], data_rec[2]))

    with open('test_set.txt', 'w') as FILE:
        for data_rec in test_set_list:
            FILE.write('%s\t%s\t%s\n'%(data_rec[0], data_rec[1], data_rec[2]))

    user_rank_dict = compute_user_rank(user_item_ratings)
    item_rank_dict = compute_item_rank(user_item_ratings)

    eta_list = [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-5, 2e-5, 3e-5, 4e-5, 7e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 0.001, 0.002, 0.003, 0.004,0.005]
    mae_list_0 = []
    mae_list_1 = []
    DME_list_0 = []
    DME_list_1 = []
    for eta in eta_list : 
        user_features, item_features = optimize_function_err_fun(train_set_dict, test_set_dict, user_rank_dict, item_rank_dict, 100, 1e-4, eta)
        (mae_0, DME_0) = predict_mf(test_set_dict, item_rank_dict.keys(), user_features, item_features)
        (mae_1, DME_1) = compute_mf(train_set_dict, test_set_dict, item_rank_dict, eta)
        mae_list_0.append(mae_0) 
        mae_list_1.append(mae_1)
        DME_list_0.append(DME_0)
        DME_list_1.append(DME_1)

    plt_0, = plt.plot(eta_list, mae_list_0)
    plt_1, = plt.plot(eta_list, mae_list_1)
    plt.legend([plt_0, plt_1], ['Zipf Matrix Factorization', 'Vanila Matrix Factorization'], loc='best')
    plt.xlabel('Zipf Penalty Coefficient')
    plt.ylabel('MAE')
    plt.show()

   
    plt_0, = plt.plot(eta_list, DME_list_0)
    plt_1, = plt.plot(eta_list, DME_list_1)
    plt.legend([plt_0, plt_1], ['Zipf Matrix Factorization', 'Vanila Matrix Factorization'], loc='best')
    plt.xlabel('Zipf Penalty Coefficient')
    plt.ylabel('Degree of Matthew Effect')
    plt.show()

