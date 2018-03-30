import numpy as np
from utils import load_ndarray_data, normalize_matrix, create_train_batchs, data_augment, create_eval_batchs, \
    score_eval_batch, score_eval_batch_map, data_augment_v2
from nn.nn_with_dropoutnet import RecommendNet as NN
from nn.nn_with_dropoutnet_v2 import RecommendNet as NN_v2

from progress.bar import ChargingBar as Bar
from prettytable import PrettyTable

import cPickle
import os

import pandas as pd
from collections import OrderedDict

from scipy.sparse import csr_matrix, lil_matrix

n_users = 627
n_items = 12

def get_score_information(lst_best_scores, lst_best_score_names, dataset_path):
    strs = ["dataset train path: %s" % dataset_path]
    for best_score, best_score_name in zip(lst_best_scores, lst_best_score_names):
        strs.extend(["%s: %.3f" % (best_score_name, best_score)])

    return "\n".join(strs)

def main_v2(u_preference_path, v_preference_path, u_bias_path, v_bias_path, u_content_path, topk,
            v_content_path, train_path, test_warm_path, n_items_per_user, batch_size, n_epoch, learning_rate, **kargs):

    u_pref_matrix = load_ndarray_data(data_path=u_preference_path, type='bin').reshape(n_users, 200)
    v_pref_matrix = load_ndarray_data(data_path=v_preference_path, type='bin').reshape(n_items, 200)
    u_bias_matrix = load_ndarray_data(data_path=u_bias_path, type='bin').reshape(n_users, 1)
    v_bias_matrix = load_ndarray_data(data_path=v_bias_path, type='bin').reshape(n_items, 1)

    u_content_matrix = load_ndarray_data(data_path=u_content_path, type='bin').reshape(n_users, 200)
    v_content_matrix = load_ndarray_data(data_path=v_content_path, type='bin').reshape(n_items, 200)

    train_data_df = pd.read_csv(train_path, sep=',',names=['profile','item','rating'])
    test_warm_data_df = pd.read_csv(test_warm_path, sep=',',names=['profile','item','rating'])
    test_warm_matrix  = test_warm_data_df.as_matrix() #test_warm_data_df.as_matrix()

    train_matrix = data_augment(R=train_data_df.as_matrix(),n_items_per_user=n_items_per_user,batch_size=batch_size,
                                u_pref=u_pref_matrix,v_pref=v_pref_matrix,u_bias=u_bias_matrix,v_bias=v_bias_matrix)

    n_step = 0
    freq_eval = 20
    best_cu_score, best_ci_score, best_w_score = -np.inf, -np.inf, -np.inf
    decay_lr_every = 100
    lr_decay = 0.95
    count = 0

    n_samples = train_matrix.shape[0]
    batch_se = [(s,min(n_samples,s + batch_size)) for s in range(0,n_samples,batch_size)]

    model = NN_v2(latent_dim=200,user_feature_dim=200,item_feature_dim=200,out_dim=200,default_lr=learning_rate,k=topk)

    cPickle.dump(model,open('./saved_model/dropoutnet_opla_vTest/model.pkl','w'))
    model.build()

    for epoch_id in range(n_epoch):
        all_loss = []

        for (s,e) in np.random.permutation(batch_se):
            count += 1
            train_batch = train_matrix[s:e,:]

            set_user = np.unique(train_batch[:,0])
            user2id = OrderedDict([(int(v),k) for k,v in enumerate(set_user)])

            row_col_ids = np.array([(user2id[r], c) for r,c in zip(train_batch[:,0],train_batch[:,1])])

            # create datas
            datas = {
                'u_pref': u_pref_matrix[user2id.keys()],
                'v_pref': v_pref_matrix,
                'u_cont': u_content_matrix[user2id.keys()],
                'v_cont': v_content_matrix,
                'u_bias': u_bias_matrix[user2id.keys()],
                'v_bias': v_bias_matrix,
                'row_col_ids': row_col_ids,
                'target': train_batch[:,2]
            }

            loss, _ = model.run(datas=datas,mode=model.train_signal)
            all_loss += [loss]

            if count % 20 == 0:
                pass

        # calculate test score
        set_user = np.unique(test_warm_matrix[:, 0])
        user2id = OrderedDict([(v, k) for k, v in enumerate(set_user)])

        row_col_ids = np.array(
            [(user2id[r], c) for r, c in zip(test_warm_matrix[:, 0], test_warm_matrix[:, 1])])

        # create datas
        datas = {
            'u_pref': u_pref_matrix[user2id.keys()],
            'v_pref': v_pref_matrix,
            'u_cont': u_content_matrix[user2id.keys()],
            'v_cont': v_content_matrix,
            'u_bias': u_bias_matrix[user2id.keys()],
            'v_bias': v_bias_matrix
        }

        target = csr_matrix((np.ones(test_warm_matrix.shape[0]), (row_col_ids[:, 0], row_col_ids[:, 1])),
                            shape=(np.max(set_user) + 1, n_items)).toarray()
        tf_eval_preds_batch, _ = model.run(datas=datas, mode=model.inf_signal)

        print ('-- test recall score: ', score(target=target, tf_eval_preds_batch=tf_eval_preds_batch))
        print ('train loss:' , np.mean(all_loss))


def score(target,tf_eval_preds_batch):

    y_nz = [sum(x) > 0 for x in target]
    y_nz = np.arange(target.shape[0])[y_nz]

    preds_all = tf_eval_preds_batch[y_nz, :]
    y = target[y_nz, :]

    x = lil_matrix(y.shape)
    x.rows = preds_all
    x.data = np.ones_like(preds_all)
    x = x.toarray()

    z = x * y
    return np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1)))

if __name__ == '__main__':
    sub_path = './data/opla'

    params = {
        'test_cold_user_path': sub_path + '/split/test_cold_user.csv',
        'test_cold_item_path': sub_path + '/split/test_cold_item.csv',
        'test_warm_path': sub_path + '/split/test_warm.csv',
        'train_path': sub_path + '/split/train.csv',
        'u_preference_path': sub_path + '/matrix_decompose/U.csv.bin',
        'v_preference_path': sub_path + '/matrix_decompose/V.csv.bin',
        'u_bias_path': sub_path + '/matrix_decompose/U_bias.csv.bin',
        'v_bias_path': sub_path + '/matrix_decompose/V_bias.csv.bin',
        'u_content_path': sub_path + '/vect/deep/profile.csv.bin',
        'v_content_path': sub_path + '/vect/deep/item.csv.bin',
        'saved_model': './saved_model/dropoutnet_opla_vTest',
        'batch_size': 100,
        'n_epoch': 20,
        'dropout': 0.3,
        'learning_rate': 0.002,
        'n_items_per_user': 8,
        'topk' : 5
    }

    main_v2(**params)
    #main(**params)