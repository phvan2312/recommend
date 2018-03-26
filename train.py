import numpy as np
from utils import load_ndarray_data, normalize_matrix, create_train_batchs, data_augment, create_eval_batchs, \
    score_eval_batch, score_eval_batch_map
from nn.nn_with_dropoutnet import RecommendNet as NN

from progress.bar import ChargingBar as Bar
from prettytable import PrettyTable

import cPickle
import os

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
    #test_cold_item_matrix = load_ndarray_data(data_path=test_cold_item_path,type='R')
    test_warm_matrix = load_ndarray_data(data_path=test_warm_path,type='R')
    train_matrix = load_ndarray_data(data_path=train_path,type='R')

    u_pref_matrix = load_ndarray_data(data_path=u_preference_path, type='bin').reshape(n_users,200)
    v_pref_matrix = load_ndarray_data(data_path=v_preference_path, type='bin').reshape(n_items,200)
    u_bias_matrix = load_ndarray_data(data_path=u_bias_path, type='bin').reshape(n_users,1)
    v_bias_matrix = load_ndarray_data(data_path=v_bias_path, type='bin').reshape(n_items,1)

    u_content_matrix = load_ndarray_data(data_path=u_content_path, type='bin').reshape(n_users,200)
    v_content_matrix = load_ndarray_data(data_path=v_content_path, type='bin').reshape(n_items,200)

    # normalize
    # u_pref_scaler, u_pref_scaled_matrix = normalize_matrix(u_pref_matrix)
    # v_pref_scaler, v_pref_scaled_matrix = normalize_matrix(v_pref_matrix)

    u_pref_scaled_matrix = u_pref_matrix
    v_pref_scaled_matrix = v_pref_matrix

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
    train_matrix = data_augment(R=train_matrix,n_items_per_user=n_items_per_user,batch_size=batch_size,
                                u_pref=u_pref_matrix,v_pref=v_pref_matrix,u_bias=u_bias_matrix,v_bias=v_bias_matrix)

    # create train and eval batchs
    train_batchs = create_train_batchs(R=train_matrix, batch_size=batch_size)

    # test_cold_user_batchs = create_eval_batchs(R=test_cold_user_matrix,batch_size=batch_size,u_pref=u_pref_scaled_matrix,
    #                                            v_pref=v_pref_scaled_matrix,u_cont=u_content_matrix,v_cont=v_content_matrix)

    # test_cold_item_batchs = create_eval_batchs(R=test_cold_item_matrix,batch_size=batch_size,u_pref=u_pref_scaled_matrix,
    #                                            v_pref=v_pref_scaled_matrix,u_cont=u_content_matrix,v_cont=v_content_matrix)

    test_warm_batchs = create_eval_batchs(R=test_warm_matrix,batch_size=batch_size,u_pref=u_pref_scaled_matrix,
                                               v_pref=v_pref_scaled_matrix,u_cont=u_content_matrix,v_cont=v_content_matrix,
                                          u_bias=u_bias_matrix, v_bias=v_bias_matrix)

    # model
    model = NN(latent_dim=200,user_feature_dim=200,item_feature_dim=200,out_dim=200,default_lr=learning_rate,
               k=topk, np_u_pref_scaled=u_pref_scaled_matrix,np_v_pref_scaled=v_pref_scaled_matrix,
               np_u_cont=u_content_matrix,np_v_cont=v_content_matrix,np_u_bias=u_bias_matrix,np_v_bias=v_bias_matrix,do_batch_norm=True)
    #model.build()

    # # save params
    # params = {
    #     'u_cont_path': saved_model + '/matrix/u_cont.bin',
    #     'v_cont_path': saved_model + '/matrix/v_cont.bin',
    #     'u_pref_path': saved_model + '/matrix/u_pref.bin',
    #     'v_pref_path': saved_model + '/matrix/v_pref.bin',
    #     'tf_model_path': saved_model + '/tf_model/dropoutnet.ckpt',
    #     'info_path': saved_model + '/info.txt',
    #     'tf_model_params': model.get_params(),
    #     'n_user': u_pref_scaled_matrix.shape[0],
    #     'n_item': v_pref_scaled_matrix.shape[0]
    # }
    #
    # u_content_matrix.tofile(open(params['u_cont_path'], 'w'))
    # v_content_matrix.tofile(open(params['v_cont_path'], 'w'))
    # u_pref_scaled_matrix.tofile(open(params['u_pref_path'], 'w'))
    # v_pref_scaled_matrix.tofile(open(params['v_pref_path'], 'w'))
    #
    # with open(saved_model + '/params.pkl', 'w') as f:
    #     cPickle.dump(params, f)

    #cPickle.dump(model, open('./saved_model/model.pkl'))
    model.build()

    # train
    batch_len = len(train_batchs)
    n_step = 0
    freq_eval = 20
    best_cu_score, best_ci_score, best_w_score = -np.inf, -np.inf, -np.inf
    decay_lr_every = 100
    lr_decay = 0.95
    #topks = [1, 2, topk]

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
                cu_acc_mean = 1.0 #score_eval_batch(datas=test_cold_user_batchs, model=model)
                ci_acc_mean = 1.0 #score_eval_batch_map(datas=test_cold_item_batchs, model=model, topks=topks)
                w_acc_mean  = score_eval_batch(datas=test_warm_batchs, model=model)

                status = '++++'

                if (cu_acc_mean + ci_acc_mean + w_acc_mean) >= (best_cu_score + best_ci_score + best_w_score):
                    best_w_score = w_acc_mean
                    best_ci_score = ci_acc_mean
                    best_cu_score = cu_acc_mean

                    status = 'best'

                    # save model
                    #model.save(params['tf_model_path'])
                    # info = get_score_information(lst_best_scores=[best_cu_score,best_ci_score,best_w_score],
                    #                              lst_best_score_names=['cold_user score', 'cold_item score', 'warm score'],
                    #                              dataset_path=train_path)
                    # with open(params['info_path'],'w') as f: f.write(info)

                tabl.add_row([status,epoch_i + 1, batch_i + 1, "%.3f" % cu_acc_mean, "%.3f" % ci_acc_mean, "%.3f" % w_acc_mean,
                              "%.4f" % (cu_acc_mean + w_acc_mean + ci_acc_mean)])

        print "\nmean_loss: %.3f" % np.mean(train_loss_total)
        print (tabl.get_string(title="Local All Test Accuracies"))

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
        'n_epoch': 30,
        'dropout': 0.0,
        'learning_rate': 0.005,
        'n_items_per_user': 8,
        'topk' : 5
    }

    main(**params)