import numpy as np
from utils import load_ndarray_data, normalize_matrix, create_train_batchs, data_augment, create_eval_batchs, \
    score_eval_batch, score_eval_batch_map
from nn.nn_with_dropoutnet import RecommendNet as NN

from progress.bar import ChargingBar as Bar
from prettytable import PrettyTable

import cPickle
import os, argparse
import pandas as pd

from lightfm import LightFM

n_users = 627 #6040
n_items = 12  #3952

prefix_save_path = './saved_model/vTest/'

parser  = argparse.ArgumentParser(description='... DROPOUTNET: RECOMMENDATION ...')

parser.add_argument('--test_cold_user_path',help='path containing data for testing cold_user case',dest='test_cold_user_path',type=str)
parser.add_argument('--test_warm_path',help='path containing data for testing warm case',dest='test_warm_path',type=str)
parser.add_argument('--train_path',help='path containing data for training', dest='train_path',type=str)
parser.add_argument('--lightfm_path',type=str,dest='lightfm_path',help='lightfm_path')
parser.add_argument('--u_content_path',help='path for storing u_content matrix',dest='u_content_path',type=str)
parser.add_argument('--v_content_path',help='path for storing v_content matrix',dest='v_content_path',type=str)
parser.add_argument('--saved_model',help='path for storing model',dest='saved_model',type=str)
parser.add_argument('--batch_size', help='number of samples per batch',dest='batch_size',type=int)
parser.add_argument('--n_epoch', help='number of epoch', dest='n_epoch',type=int)
parser.add_argument('--dropout', help='dropout keep probability', dest='dropout',type=float)
parser.add_argument('--learning_rate', help='learning rate', dest='learning_rate',type=float)
parser.add_argument('--n_items_per_user', help='number of items to be selected per user', dest='n_items_per_user',type=int)
parser.add_argument('--topk', help='number of items to be considered (top-k)',dest='topk',type=int)
parser.add_argument('--latent_dim', help='latent dimension', dest='latent_dim', type=int)
parser.add_argument('--content_dim', help='content dimension', dest='content_dim', type=int)

args    = vars(parser.parse_args())

"""
USAGE: assume we're in recommend folder
python train.py --test_cold_user_path=./data/opla/split/test_cold_user.csv --test_warm_path=./data/opla/split/test_warm.csv
--train_path=./data/opla//split/train.csv --u_content_path=./data/opla/vect/deep/profile.csv.bin 
--v_content_path=./data/opla/vect/deep/item.csv.bin --saved_model=./saved_model/dropoutnet_opla_vTest_2 --batch_size=100 --n_epoch=20
--dropout=0.5 --learning_rate=0.001 --n_items_per_user=8 --topk=6 --latent_dim=200 --content_dim=200 --lightfm_path=./data/opla/matrix_decompose/lightfm.pkl

"""

def get_score_information(lst_best_scores, lst_best_score_names, dataset_path):
    strs = ["dataset train path: %s" % dataset_path]
    for best_score, best_score_name in zip(lst_best_scores, lst_best_score_names):
        strs.extend(["%s: %.3f" % (best_score_name, best_score)])

    return "\n".join(strs)

def main(test_cold_user_path, test_warm_path, train_path, u_content_path, v_content_path, saved_model, batch_size, n_epoch, learning_rate,
         n_items_per_user, dropout, topk, latent_dim, content_dim, lightfm_path, **kargs):

    test_cold_user_matrix = pd.read_csv(test_cold_user_path,sep=',',names=['profile','item','rating'])
    test_warm_matrix = pd.read_csv(test_warm_path,sep=',',names=['profile','item','rating'])
    train_matrix = pd.read_csv(train_path,sep=',',names=['profile','item','rating'])

    lightfm = cPickle.load(open(lightfm_path, 'r'))

    u_pref_matrix = lightfm.user_embeddings.astype('float32')
    v_pref_matrix = lightfm.item_embeddings.astype('float32')
    u_bias_matrix = lightfm.user_biases.reshape((-1, 1)).astype('float32')
    v_bias_matrix = lightfm.item_biases.reshape((-1, 1)).astype('float32')

    u_content_matrix = np.fromfile(file=u_content_path,dtype=np.float32).reshape(n_users,content_dim)
    v_content_matrix = np.fromfile(file=v_content_path,dtype=np.float32).reshape(n_items,content_dim)

    # adding zero for dropout purpose
    u_pref_zero_matrix = np.vstack([u_pref_matrix, np.zeros_like(u_pref_matrix[0,:])])
    v_pref_zero_matrix = np.vstack([v_pref_matrix, np.zeros_like(v_pref_matrix[0,:])])
    u_bias_zero_matrix = np.vstack([u_bias_matrix, np.zeros_like(u_bias_matrix[0,:])])
    v_bias_zero_matrix = np.vstack([v_bias_matrix, np.zeros_like(v_bias_matrix[0,:])])

    zero_user_id = u_pref_zero_matrix.shape[0] - 1
    zero_item_id = v_pref_zero_matrix.shape[0] - 1

    # create model for storing necessary materials
    if not os.path.isdir(saved_model): os.makedirs(saved_model)
    save_paths = {
        'model_class': os.path.join(saved_model, 'model.pkl'),
        'model_deep': os.path.join(saved_model, 'model.ckpt')
    }

    # augment training datas
    train_matrix = data_augment(R=train_matrix,n_items_per_user=n_items_per_user,batch_size=batch_size,
                                u_pref=u_pref_matrix,v_pref=v_pref_matrix,u_bias=u_bias_matrix,v_bias=v_bias_matrix)

    # create train and eval batchs
    train_batchs = create_train_batchs(R=train_matrix, batch_size=batch_size)
    test_cold_user_batchs = create_eval_batchs(R=test_cold_user_matrix,batch_size=batch_size,is_cold_user=True)
    test_warm_batchs = create_eval_batchs(R=test_warm_matrix,batch_size=batch_size)

    # model
    model = NN(latent_dim=latent_dim,user_feature_dim=content_dim,item_feature_dim=content_dim,out_dim=200,
               default_lr=learning_rate, k=topk,do_batch_norm=True)

    cPickle.dump(model, open(save_paths['model_class'],'w'))
    model.set_np_matrix(np_u_pref_zero=u_pref_zero_matrix,np_v_pref_zero=v_pref_zero_matrix,np_u_cont=u_content_matrix,
                        np_v_cont=v_content_matrix,np_u_bias_zero=u_bias_zero_matrix,np_v_bias_zero=v_bias_zero_matrix)
    model.build()

    # train
    batch_len = len(train_batchs)
    n_step = 0
    freq_eval = 40
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

                tabl.add_row([status,epoch_i + 1, batch_i + 1, "%.3f" % cu_acc_mean, "%.3f" % ci_acc_mean, "%.3f" % w_acc_mean,
                              "%.4f" % (cu_acc_mean + w_acc_mean + ci_acc_mean)])

        print "\nmean_loss: %.3f" % np.mean(train_loss_total)
        print (tabl.get_string(title="Local All Test Accuracies"))

if __name__ == '__main__':
    sub_path = './fake_data/opla'

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
        'saved_model': './saved_model/dropoutnet_opla_vTest_2',
        'batch_size': 100,
        'n_epoch': 20,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'n_items_per_user': 8,
        'topk' : 6,
        'latent_dim' : 200
    }

    # sub_path = './fake_data/movielen1m'
    #
    # params = {
    #     'test_cold_user_path': sub_path + '/fake_data/test_cold_user.csv',
    #     'test_cold_item_path': sub_path + '/fake_data/test_cold_item.csv',
    #     'test_warm_path': sub_path + '/fake_data/test_warm.csv',  # sub_path + '/split/test_warm.csv',
    #     'train_path': sub_path + '/fake_data/train.csv',  # sub_path + '/split/train.csv',
    #     'u_preference_path': './fake_data/opla' + '/matrix_decompose/U.csv.bin',  # './fake_data/data2/train/user_features_0based.txt',
    #     'v_preference_path': './fake_data/opla' + '/matrix_decompose/V.csv.bin',  # './fake_data/data2/train/item_features_0based.txt',
    #     'u_bias_path': './fake_data/opla' + '/matrix_decompose/U_bias.csv.bin',
    #     'v_bias_path': './fake_data/opla' + '/matrix_decompose/V_bias.csv.bin',
    #     'u_content_path': sub_path + '/fake_data/user_feature.csv.bin',  # sub_path + '/vect/deep/profile.csv.bin',
    #     'v_content_path': sub_path + '/fake_data/item_feature.csv.bin',  # sub_path + '/vect/deep/item.csv.bin',
    #     'saved_model': './saved_model/dropoutnet_opla_vTest_2',
    #     'batch_size': 512,
    #     'n_epoch': 20,
    #     'dropout': 0.5,
    #     'learning_rate': 0.005,
    #     'n_items_per_user': 200,
    #     'topk': 10
    # }

    #main(**params)
    main(**args)