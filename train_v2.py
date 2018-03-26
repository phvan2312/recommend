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

def main(test_cold_user_path, test_cold_item_path, test_warm_path, train_path, u_preference_path, v_preference_path, u_bias_path, v_bias_path, u_content_path,
         v_content_path, saved_model, batch_size, n_epoch, learning_rate, n_items_per_user, dropout, topk, **kargs):

    test_cold_user_matrix = load_ndarray_data(data_path=test_cold_user_path,type='R')
    test_warm_matrix = load_ndarray_data(data_path=test_warm_path,type='R')
    train_matrix = load_ndarray_data(data_path=train_path,type='R')

    u_pref_matrix = load_ndarray_data(data_path=u_preference_path, type='bin').reshape(n_users,200)
    v_pref_matrix = load_ndarray_data(data_path=v_preference_path, type='bin').reshape(n_items,200)
    u_bias_matrix = load_ndarray_data(data_path=u_bias_path, type='bin').reshape(n_users,1)
    v_bias_matrix = load_ndarray_data(data_path=v_bias_path, type='bin').reshape(n_items,1)

    u_content_matrix = load_ndarray_data(data_path=u_content_path, type='bin').reshape(n_users,200)
    v_content_matrix = load_ndarray_data(data_path=v_content_path, type='bin').reshape(n_items,200)

    # normalize
    u_pref_scaler, u_pref_scaled_matrix = normalize_matrix(u_pref_matrix)
    v_pref_scaler, v_pref_scaled_matrix = normalize_matrix(v_pref_matrix)

    # adding zero for dropout purpose
    u_pref_scaled_matrix = np.vstack([u_pref_scaled_matrix, np.zeros_like(u_pref_scaled_matrix[0,:])])
    v_pref_scaled_matrix = np.vstack([v_pref_scaled_matrix, np.zeros_like(v_pref_scaled_matrix[0,:])])
    zero_user_id = u_pref_scaled_matrix.shape[0] - 1
    zero_item_id = v_pref_scaled_matrix.shape[0] - 1

    # save u,v content_matrix, u_pref_scaled_matrix, v_pref_scaled_matrix
    # TODO: need a mapping from real user

    if not os.path.exists(saved_model + '/matrix'):
        os.makedirs(saved_model + '/matrix')

    if not os.path.exists(saved_model + '/tf_model'):
        os.makedirs(saved_model + '/tf_model')

    # augment training datas
    # train_matrix = data_augment(R=train_matrix,n_items_per_user=n_items_per_user,batch_size=batch_size,
    #                             u_pref=u_pref_matrix,v_pref=v_pref_matrix,u_bias=u_bias_matrix,v_bias=v_bias_matrix)

    train_matrix = data_augment_v2(R=pd.read_csv(train_path,sep=',',names=['profile','item','rating']), n_items_per_user=n_items_per_user, batch_size=batch_size)

    # create train and eval batchs
    train_batchs = create_train_batchs(R=train_matrix, batch_size=batch_size)

    test_cold_user_batchs = create_eval_batchs(R=test_cold_user_matrix,batch_size=batch_size,u_pref=u_pref_scaled_matrix,
                                               v_pref=v_pref_scaled_matrix,u_cont=u_content_matrix,v_cont=v_content_matrix)

    test_warm_batchs = create_eval_batchs(R=test_warm_matrix,batch_size=batch_size,u_pref=u_pref_scaled_matrix,
                                               v_pref=v_pref_scaled_matrix,u_cont=u_content_matrix,v_cont=v_content_matrix)

    # model
    model = NN(latent_dim=200,user_feature_dim=200,item_feature_dim=200,out_dim=200,default_lr=learning_rate,
               k=topk, np_u_pref_scaled=u_pref_scaled_matrix,np_v_pref_scaled=v_pref_scaled_matrix,
               np_u_cont=u_content_matrix,np_v_cont=v_content_matrix)
    #model.build()

    cPickle.dump(model, open('./saved_model/model.pkl'))
    model.build()

    # train
    batch_len = len(train_batchs)
    n_step = 0
    freq_eval = 20
    best_cu_score, best_ci_score, best_w_score = -np.inf, -np.inf, -np.inf
    decay_lr_every = 100
    lr_decay = 0.95

    for epoch_i in range(n_epoch):
        print ("\n[E]poch %i, lr %.4f, dropout %.2f" % (epoch_i + 1, learning_rate, dropout))
        bar = Bar('training',max=batch_len,suffix='')
        tabl = PrettyTable(['status','train_epoch', 'train_batch', 'cold_user_acc', 'cold_item_acc', 'warm_acc', 'total'])

        batch_ids = np.random.permutation(batch_len)
        train_loss_total = []

        for batch_i, batch_id in enumerate(batch_ids):
            n_step += 1
            batch = train_batchs[batch_id]

            # dropout or not
            if dropout != 0:
                batch.dropout_user(dropout_prob = dropout, zero_id = zero_user_id)
                batch.dropout_item(dropout_prob = dropout, zero_id = zero_item_id)

            train_loss, _ = model.run(datas=batch, mode=model.train_signal, lr = learning_rate)
            train_loss_total.append(train_loss)

            bar.bar_prefix = " | batch %i |" % (batch_i + 1)
            bar.bar_suffix = " | cur_loss: %.4f | " % train_loss

            bar.next()

            if n_step % decay_lr_every == 0:
                learning_rate = lr_decay * learning_rate

            if n_step % freq_eval == 0:
                cu_acc_mean = score_eval_batch(datas=test_cold_user_batchs, model=model)
                ci_acc_mean = 1.0 #score_eval_batch_map(datas=test_cold_item_batchs, model=model, topks=topks)
                w_acc_mean  = score_eval_batch(datas=test_warm_batchs, model=model)

                status = '++++'

                if (cu_acc_mean + ci_acc_mean + w_acc_mean) >= (best_cu_score + best_ci_score + best_w_score):
                    best_w_score = w_acc_mean
                    best_ci_score = ci_acc_mean
                    best_cu_score = cu_acc_mean

                    status = 'best'

                    # save model
                    model.save(params['tf_model_path'])
                    info = get_score_information(lst_best_scores=[best_cu_score,best_ci_score,best_w_score],
                                                 lst_best_score_names=['cold_user score', 'cold_item score', 'warm score'],
                                                 dataset_path=train_path)
                    with open(params['info_path'],'w') as f: f.write(info)

                tabl.add_row([status,epoch_i + 1, batch_i + 1, "%.3f" % cu_acc_mean, "%.3f" % ci_acc_mean, "%.3f" % w_acc_mean,
                              "%.4f" % (cu_acc_mean + w_acc_mean + ci_acc_mean)])

        print "\nmean_loss: %.3f" % np.mean(train_loss_total)
        print (tabl.get_string(title="Local All Test Accuracies"))

def main_v2(u_preference_path, v_preference_path, u_bias_path, v_bias_path, u_content_path,
            v_content_path, train_path, test_warm_path, n_items_per_user, batch_size, n_epoch, learning_rate, **kargs):

    topk = 5

    u_pref_matrix = load_ndarray_data(data_path=u_preference_path, type='bin').reshape(n_users, 200)
    v_pref_matrix = load_ndarray_data(data_path=v_preference_path, type='bin').reshape(n_items, 200)
    u_bias_matrix = load_ndarray_data(data_path=u_bias_path, type='bin').reshape(n_users, 1)
    v_bias_matrix = load_ndarray_data(data_path=v_bias_path, type='bin').reshape(n_items, 1)

    u_content_matrix = load_ndarray_data(data_path=u_content_path, type='bin').reshape(n_users, 200)
    v_content_matrix = load_ndarray_data(data_path=v_content_path, type='bin').reshape(n_items, 200)

    train_data_df = pd.read_csv(train_path, sep=',',names=['profile','item','rating'])
    test_warm_data_df = pd.read_csv(test_warm_path, sep=',',names=['profile','item','rating'])
    test_warm_matrix  = test_warm_data_df.as_matrix() #test_warm_data_df.as_matrix()

    # test_row, test_col, test_rating = test_warm_data_df['profile'].values, test_warm_data_df['item'].values, \
    #                                   test_warm_data_df['rating'].values
    # test_warm_matrix = csr_matrix((test_rating,(test_row, test_col)),shape=(n_users, n_items)).toarray()

    #train_matrix = data_augment_v2(R=train_data_df,n_items_per_user=n_items_per_user,batch_size=batch_size)
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