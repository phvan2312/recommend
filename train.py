import numpy as np
from utils import load_ndarray_data, normalize_matrix, create_train_batchs, data_augment, create_eval_batchs, \
    score_eval_batch, score_eval_batch_map
from nn.nn_with_dropoutnet import RecommendNet as NN

from progress.bar import ChargingBar as Bar
from prettytable import PrettyTable

import cPickle
import os
import pandas as pd

n_users = 6040
n_items = 3952

prefix_save_path = './saved_model/vTest/'
save_paths = {
    'model_class' : os.path.join(prefix_save_path,'model.pkl'),
    'model_deep'  : os.path.join(prefix_save_path,'model.ckpt')
}

def get_score_information(lst_best_scores, lst_best_score_names, dataset_path):
    strs = ["dataset train path: %s" % dataset_path]
    for best_score, best_score_name in zip(lst_best_scores, lst_best_score_names):
        strs.extend(["%s: %.3f" % (best_score_name, best_score)])

    return "\n".join(strs)

def main(test_cold_user_path, test_cold_item_path, test_warm_path, train_path, u_preference_path, v_preference_path,
         u_bias_path, v_bias_path, u_content_path, v_content_path, saved_model, batch_size, n_epoch, learning_rate,
         n_items_per_user, dropout, topk, **kargs):

    test_cold_user_matrix = pd.read_csv(test_cold_user_path,sep=',',names=['profile','item','rating']) #load_ndarray_data(data_path=test_cold_user_path,type='R')
    #test_cold_item_matrix = load_ndarray_data(data_path=test_cold_item_path,type='R')
    test_warm_matrix = pd.read_csv(test_warm_path,sep=',',names=['profile','item','rating']) #load_ndarray_data(data_path=test_warm_path,type='R')
    train_matrix = pd.read_csv(train_path,sep=',',names=['profile','item','rating']) #load_ndarray_data(data_path=train_path,type='R')

    u_pref_matrix = np.fromfile(file=u_preference_path,dtype=np.float32).reshape(n_users,200)
    v_pref_matrix = np.fromfile(file=v_preference_path,dtype=np.float32).reshape(n_items,200)
    u_bias_matrix = np.fromfile(file=u_bias_path,dtype=np.float32).reshape(n_users,1)
    v_bias_matrix = np.fromfile(file=v_bias_path,dtype=np.float32).reshape(n_items,1)

    u_content_matrix = np.fromfile(file=u_content_path,dtype=np.float32).reshape(n_users,30)
    v_content_matrix = np.fromfile(file=v_content_path,dtype=np.float32).reshape(n_items,18)

    # adding zero for dropout purpose
    u_pref_zero_matrix = np.vstack([u_pref_matrix, np.zeros_like(u_pref_matrix[0,:])])
    v_pref_zero_matrix = np.vstack([v_pref_matrix, np.zeros_like(v_pref_matrix[0,:])])
    u_bias_zero_matrix = np.vstack([u_bias_matrix, np.zeros_like(u_bias_matrix[0,:])])
    v_bias_zero_matrix = np.vstack([v_bias_matrix, np.zeros_like(v_bias_matrix[0,:])])

    zero_user_id = u_pref_zero_matrix.shape[0] - 1
    zero_item_id = v_pref_zero_matrix.shape[0] - 1

    # if not os.path.exists(saved_model + '/matrix'): os.makedirs(saved_model + '/matrix')
    # if not os.path.exists(saved_model + '/tf_model'): os.makedirs(saved_model + '/tf_model')

    # augment training datas
    train_matrix = data_augment(R=train_matrix,n_items_per_user=n_items_per_user,batch_size=batch_size,
                                u_pref=u_pref_matrix,v_pref=v_pref_matrix,u_bias=u_bias_matrix,v_bias=v_bias_matrix)

    # create train and eval batchs
    train_batchs = create_train_batchs(R=train_matrix, batch_size=batch_size)

    test_cold_user_batchs = create_eval_batchs(R=test_cold_user_matrix,batch_size=batch_size,is_cold_user=True)

    # test_cold_item_batchs = create_eval_batchs(R=test_cold_item_matrix,batch_size=batch_size,u_pref=u_pref_zero_matrix,
    #                                            v_pref=v_pref_zero_matrix,u_cont=u_content_matrix,v_cont=v_content_matrix)

    test_warm_batchs = create_eval_batchs(R=test_warm_matrix,batch_size=batch_size)

    # model
    model = NN(latent_dim=200,user_feature_dim=30,item_feature_dim=18,out_dim=200,default_lr=learning_rate, k=topk,do_batch_norm=True)

    cPickle.dump(model, open(save_paths['model_class'],'w'))
    model.set_np_matrix(np_u_pref_zero=u_pref_zero_matrix,np_v_pref_zero=v_pref_zero_matrix,np_u_cont=u_content_matrix,
                        np_v_cont=v_content_matrix,np_u_bias_zero=u_bias_zero_matrix,np_v_bias_zero=v_bias_zero_matrix)
    model.build()

    # train
    batch_len = len(train_batchs)
    n_step = 0
    freq_eval = 2000
    best_cu_score, best_ci_score, best_w_score = -np.inf, -np.inf, -np.inf
    decay_lr_every = 1000
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
                #batch.dropout_item(dropout_prob = dropout, zero_id = zero_item_id)

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
                    model.save(save_paths['model_deep'])
                    # info = get_score_information(lst_best_scores=[best_cu_score,best_ci_score,best_w_score],
                    #                              lst_best_score_names=['cold_user score', 'cold_item score', 'warm score'],
                    #                              dataset_path=train_path)
                    # with open(params['info_path'],'w') as f: f.write(info)

                tabl.add_row([status,epoch_i + 1, batch_i + 1, "%.3f" % cu_acc_mean, "%.3f" % ci_acc_mean, "%.3f" % w_acc_mean,
                              "%.4f" % (cu_acc_mean + w_acc_mean + ci_acc_mean)])

        print "\nmean_loss: %.3f" % np.mean(train_loss_total)
        print (tabl.get_string(title="Local All Test Accuracies"))

if __name__ == '__main__':
    # sub_path = './data/opla'
    #
    # params = {
    #     'test_cold_user_path': sub_path + '/split/test_cold_user.csv',
    #     'test_cold_item_path': sub_path + '/split/test_cold_item.csv',
    #     'test_warm_path': './data/data2/train/test/test_warm.csv',#sub_path + '/split/test_warm.csv',
    #     'train_path': './data/data2/train/train/train.csv',#sub_path + '/split/train.csv',
    #     'u_preference_path': sub_path + '/matrix_decompose/U.csv.bin', #'./data/data2/train/user_features_0based.txt',
    #     'v_preference_path': sub_path + '/matrix_decompose/V.csv.bin', #'./data/data2/train/item_features_0based.txt',
    #     'u_bias_path': sub_path + '/matrix_decompose/U_bias.csv.bin',
    #     'v_bias_path': sub_path + '/matrix_decompose/V_bias.csv.bin',
    #     'u_content_path': './data/data2/user.csv.bin',#sub_path + '/vect/deep/profile.csv.bin',
    #     'v_content_path': './data/data2/item.csv.bin',#sub_path + '/vect/deep/item.csv.bin',
    #     'saved_model': './saved_model/dropoutnet_opla_vTest',
    #     'batch_size': 512,
    #     'n_epoch': 20,
    #     'dropout': 0.5,
    #     'learning_rate': 0.005,
    #     'n_items_per_user': 200,
    #     'topk' : 10
    # }

    sub_path = './data/movielen1m'

    params = {
        'test_cold_user_path': sub_path + '/data/test_cold_user.csv',
        'test_cold_item_path': sub_path + '/data/test_cold_item.csv',
        'test_warm_path': sub_path + '/data/test_warm.csv',  # sub_path + '/split/test_warm.csv',
        'train_path': sub_path + '/data/train.csv',  # sub_path + '/split/train.csv',
        'u_preference_path': './data/opla' + '/matrix_decompose/U.csv.bin',  # './data/data2/train/user_features_0based.txt',
        'v_preference_path': './data/opla' + '/matrix_decompose/V.csv.bin',  # './data/data2/train/item_features_0based.txt',
        'u_bias_path': './data/opla' + '/matrix_decompose/U_bias.csv.bin',
        'v_bias_path': './data/opla' + '/matrix_decompose/V_bias.csv.bin',
        'u_content_path': sub_path + '/data/user_feature.csv.bin',  # sub_path + '/vect/deep/profile.csv.bin',
        'v_content_path': sub_path + '/data/item_feature.csv.bin',  # sub_path + '/vect/deep/item.csv.bin',
        'saved_model': './saved_model/dropoutnet_opla_vTest',
        'batch_size': 512,
        'n_epoch': 20,
        'dropout': 0.5,
        'learning_rate': 0.005,
        'n_items_per_user': 200,
        'topk': 10
    }

    main(**params)