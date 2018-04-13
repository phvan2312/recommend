import cPickle

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from sklearn.utils import shuffle

import utils
from nn.isgd import ISGD


def train_continue(matrix_path,save_model_path,user2id_path,n_epochs,test_matrix_path, **kargs):
    user2id_df = pd.read_csv(user2id_path)  # 2 columns (user_name,id)
    user2id = {u: i for u, i in zip(user2id_df['user_name'].tolist(), user2id_df['id'].tolist())}

    # load matrix for training
    matrix_df = pd.read_csv(matrix_path)  # 2 columns (user_name_1, user_name_2), which means user_name_1 likes user_name_2
    for col in ['user_name_1', 'user_name_2']:
        matrix_df[col] = map(lambda x: user2id[x], matrix_df[col].tolist())

    # load matrix for testing
    test_matrix_df = pd.read_csv(test_matrix_path)  # 2 columns (user_name_1, user_name_2), which means user_name_1 likes user_name_2
    for col in ['user_name_1', 'user_name_2']:
        test_matrix_df[col] = map(lambda x: user2id[x], test_matrix_df[col].tolist())

    # training
    isgd = cPickle.load(open(save_model_path,'r'))
    for n_epoch in xrange(n_epochs):
        for i,r in matrix_df.iterrows():
            user_1, user_2 = r['user_name_1'],r['user_name_2']
            isgd.update(user_1,user_2)

    # testing
    target_matrix = coo_matrix(
        (
            np.ones(test_matrix_df.shape[0]),
            (test_matrix_df['user_name_1'].tolist(), test_matrix_df['user_name_2'].tolist())
        ),
        shape=(len(isgd.known_users), len(isgd.known_items))
    ).toarray()

    select_user_ids = [id for id in test_matrix_df['user_name_1'].tolist() if id in isgd.known_users]
    predict_item_ids = isgd.recommend(u_index=select_user_ids, N=10)

    """
    x = scipy.sparse.lil_matrix(y.shape)
        x.rows = preds_all_topk
        x.data = np.ones_like(preds_all_topk)
        x = x.toarray()
    """

    predict_matrix = lil_matrix(shape=(len(isgd.known_users), len(isgd.known_items)))
    predict_matrix.rows = predict_item_ids
    predict_matrix.data = np.ones_like(predict_item_ids)
    predict_matrix = predict_matrix.toarray()

    print utils.my_eval(target_matrix, predict_matrix)

    return isgd

def main(matrix_path,user2id_path,k,learning_rate,n_epochs,test_matrix_path):
    user2id_df = pd.read_csv(user2id_path)  # 2 columns (user_name,id)
    user2id = {u: i for u, i in zip(user2id_df['user_name'].tolist(), user2id_df['id'].tolist())}

    matrix_df = pd.read_csv(matrix_path) # 2 columns (user_name_1, user_name_2), which means user_name_1 likes user_name_2
    for col in ['user_name_1','user_name_2']:
        matrix_df[col] = map(lambda x: user2id[x], matrix_df[col].tolist())

    # load matrix for testing
    test_matrix_df = pd.read_csv(
        test_matrix_path)  # 2 columns (user_name_1, user_name_2), which means user_name_1 likes user_name_2
    for col in ['user_name_1', 'user_name_2']:
        test_matrix_df[col] = map(lambda x: user2id[x], test_matrix_df[col].tolist())

    max_user_1 = np.max(matrix_df['user_name_1'].values) + 1
    max_user_2 = np.max(matrix_df['user_name_2'].values) + 1
    max_user = np.max([max_user_1, max_user_2])

    isgd = ISGD(n_user=max_user,n_item=max_user,k=k,learning_rate=learning_rate)

    for n_epoch in xrange(n_epochs):
        epoch_errs = []
        for i,r in shuffle(matrix_df).iterrows():
            user_1, user_2 = r['user_name_1'],r['user_name_2']
            err = isgd.update(user_1,user_2)

            epoch_errs += [err]

        print ('epoch: %d, error: %.4f ' % (n_epoch + 1, np.mean(epoch_errs)))

    # testing
    target_matrix = coo_matrix(
        (
            np.ones(test_matrix_df.shape[0]),
            (test_matrix_df['user_name_1'].tolist(), test_matrix_df['user_name_2'].tolist())
        ),
        shape=(np.max(user2id.keys()) + 1, np.max(user2id.values()) + 1)
    ).toarray()

    select_user_ids = list(set([id for id in test_matrix_df['user_name_1'].tolist() if id in isgd.known_users]))
    N = 10
    predict_item_ids = np.asarray([isgd.recommend(u_index=select_user_id, N=N) for select_user_id in select_user_ids],dtype='int32')

    row_ids = np.asarray([[r] * N for r in select_user_ids],dtype='int32').reshape(-1)
    col_ids = predict_item_ids.reshape(-1)

    predict_matrix = coo_matrix(
        (
            np.ones(len(row_ids)),
            (row_ids, col_ids)
        ),
        shape=(np.max(user2id.keys()) + 1, np.max(user2id.values()) + 1)
    ).toarray()

    # predict_matrix = coo_matrix(
    #     (
    #         np.ones(test_matrix_df.shape[0]),
    #         (test_matrix_df['user_name_1'].tolist(), test_matrix_df['user_name_2'].tolist())
    #     ),
    #     shape=(np.max(user2id.keys()) + 1, np.max(user2id.values()) + 1)
    # ).toarray()

    #
    # predict_matrix = lil_matrix(np.max(user2id.keys()) + 1, np.max(user2id.values()) + 1)
    # predict_matrix.rows = predict_item_ids
    # predict_matrix.data = np.ones_like(predict_item_ids)
    # predict_matrix = predict_matrix.toarray()

    print utils.my_eval(target_matrix, predict_matrix)

    return isgd

matrix_path = './data/matrix.txt' # 2 columns (user_name_1, user_name_2), which means user_name_1 likes user_name_2
user2id_path = './data/user2id.txt' # 2 columns (user_name,id)
test_matrix_path = './data/matrix.txt'

if __name__ == '__main__':

    params = {
        'matrix_path' : matrix_path,
        'test_matrix_path' : test_matrix_path,
        'user2id_path' : user2id_path,
        'k' : 100,
        'learning_rate' : 0.01,
        'n_epochs': 10
    }

    main(**params)

    pass