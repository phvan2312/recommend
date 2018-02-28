import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from batch import TrainBatchSample, EvalBatchSample
from pymf.nmf import NMF
import scipy

def get_top_k(matrix,k):
    """
    get top k largest elements from each corressponding rows of matrix
    :param matrix: ndarray with two dimensions
    :param k: number of elements to be taken
    :return: <list_topk_values, list_topk_row_inds, list_topk_col_inds>
    """
    assert k <=  matrix.shape[1]
    col_inds = np.argpartition(matrix, -k)[:,-k:].flatten()
    row_inds = np.repeat(range(matrix.shape[0]),k)

    vals = matrix[row_inds, col_inds]

    return vals, col_inds

def get_random_k(matrix, k):
    """
    get k random element from each corressponding rows of matrix
    :param matrix: ndarray with two dimensions
    :param k: number of elements to be taken
    :return: <list_topk_values, list_topk_row_inds, list_topk_col_inds>
    """
    assert k <= matrix.shape[1]
    col_inds = np.array([np.random.choice(matrix.shape[1],k) for _ in range(matrix.shape[0])]).flatten()
    row_inds = np.repeat(range(matrix.shape[0]),k)

    vals = matrix[row_inds, col_inds]

    return vals, col_inds

def data_augment(R,n_items_per_user,batch_size, u_pref, v_pref):
    """
    :param R: training rating matrix
    :param n_items_per_user: each user will be trained with 2*n_items_per_user items
    :param batch_size: user batch size
    :param u_pref: U matrix preference
    :param v_pref: V matrix preference
    :return: R with more samples
    """
    assert n_items_per_user < v_pref.shape[0]

    new_R = {
        'uid'  : [],
        'iid'  : [],
        'rating':[]
    }

    unique_user_ids = np.unique(R[:,0])
    n_user = len(unique_user_ids)

    user_batch_sizes = [(s,min(s + batch_size, n_user)) for s in range(0,n_user,batch_size)]

    for (s,e) in user_batch_sizes:
        user_ids = unique_user_ids[s:e]

        sub_R = np.dot(u_pref[user_ids, :], v_pref.T)
        topk_u_ids = rand_u_ids = np.repeat(user_ids,n_items_per_user)

        topk_ratings, topk_i_ids = get_top_k(matrix=sub_R,k=n_items_per_user)
        rand_ratings, rand_i_ids = get_random_k(matrix=sub_R,k=n_items_per_user)

        fn_ratings = np.append(topk_ratings, rand_ratings)
        fn_u_ids   = np.append(topk_u_ids, rand_u_ids)
        fn_i_ids   = np.append(topk_i_ids, rand_i_ids)

        new_R['iid'].extend(fn_i_ids)
        new_R['uid'].extend(fn_u_ids)
        new_R['rating'].extend(fn_ratings)

    return pd.DataFrame(new_R, columns=['uid', 'iid', 'rating']).as_matrix()

def create_eval_batchs(R, batch_size, u_pref, v_pref, u_cont, v_cont):
    eval_batch = EvalBatchSample(user_ids=R[:,0],item_ids=R[:,1],ratings=R[:,2],batch_size=batch_size, u_pref=u_pref,
                                 v_pref=v_pref,u_cont=u_cont,v_cont=v_cont)
    return eval_batch

def create_train_batchs(R, batch_size, shuffle=False):
    """
    :param R: rating matrix, ndarray type, with 3 columns <uid, iid, rating>
    :param batch_size: integer
    :return: list of batch
    """
    data_len = R.shape[0]

    if shuffle: np.random.shuffle(R)

    batch_sizes = [(batch_start,min(batch_start + batch_size, data_len)) for batch_start in range(0,data_len,batch_size)]
    batchs = []
    for (batch_start, batch_end) in batch_sizes:
        user_ids = R[batch_start:batch_end, 0]
        item_ids = R[batch_start:batch_end, 1]
        ratings  = R[batch_start:batch_end, 2]

        batch = TrainBatchSample(user_ids=user_ids, item_ids=item_ids, ratings=ratings)
        batchs.append(batch)

    return batchs

def load_ndarray_data(data_path, type='bin'):
    """
    :param data_path: path to data
    :param type: must be one of the following ['bin','svmlight','R']
    :return: ndarray
    """
    types = ['bin','svmlight','R']
    assert type in types

    if type == 'bin':
        res = np.fromfile(file=data_path,dtype=np.float32)
    elif type == 'svmlight':
        res, _ = datasets.load_svmlight_file(data_path, zero_based=True, dtype=np.float32)
        res = res.toarray()
    elif type == 'R':
        df = pd.read_csv(data_path,sep=',',names=['uid','iid','rating'])
        res = df.as_matrix()
    else:
        raise Exception('Type does not exists')

    return res

def normalize_matrix(data,scaler_class = StandardScaler):
    """
    :param data: data to be normalized
    :param scaler_class: default is StandardScaler
    :return: <scaler, normalized data>
    """
    scaler = scaler_class()
    scaler.fit(data)

    norm_data = scaler.transform(data)

    norm_data[norm_data > 5.0] = 5.0
    norm_data[norm_data < -5.0] = -5.0
    norm_data[np.absolute(norm_data) < 1e-5] = 0.0

    return scaler, norm_data

def get_ids_rated_by_x(R,x_id,x_col_id):
    ids = np.where(R[:,x_col_id] == x_id)[0]

    return ids

def matrix_decomposite(R,k=200,n_iter=200):
    """
    :param R: rating matrix
    :param k: latent dim
    :param n_iter: number of iterations for algorithm
    :return: U,V
    """
    n_users = np.max(R[:,0]) + 1
    n_items = np.max(R[:,1]) + 1

    new_R = np.zeros(shape=(n_users,n_items),dtype='float32')
    new_R[R[:,0].tolist(),R[:,1].tolist()] = R[:,2].astype('float32')

    mdl = NMF(new_R,num_bases=k)
    mdl.factorize(niter=n_iter)

    U,V = mdl.W, mdl.H.T

    return U,V

def extract_warm_from_R(R,split_percent=0.1):
    """
    :param R: rating matrix
    :param split_percent: indicate number of samples to be extracted for each user
    :return: test warm rating matrix (R)
    """
    user_ids = np.unique(R[:, 0]).tolist()

    final_i_ids = []
    final_u_ids = []
    final_ratings = []
    final_ids = []

    # get from each users some items
    for user_id in user_ids:
        ids = get_ids_rated_by_x(R, x_id=user_id, x_col_id=0)
        rand_len = int(split_percent * ids.shape[0])
        select_ids = ids[np.random.permutation(ids.shape[0])][:rand_len]

        select_i_ids = R[select_ids,1]
        select_u_ids = R[select_ids,0]
        select_ratings = R[select_ids,2]

        if len(select_ids) > 0: final_ids.extend(select_ids.tolist())
        if len(select_i_ids) > 0: final_i_ids.extend(select_i_ids.tolist())
        if len(select_u_ids) > 0: final_u_ids.extend(select_u_ids.tolist())
        if len(select_ratings) > 0: final_ratings.extend(select_ratings.tolist())

    return pd.DataFrame(data={'uid': final_u_ids, 'iid': final_i_ids, 'rating': final_ratings},
                        columns=['uid', 'iid', 'rating']).as_matrix(), final_ids

def extract_cold_user_from_R(R, split_percent=0.1):
    """
    :param R: rating matrix
    :param split_percent: indicate number of samples to be extracted
    :return: test cold_user rating matrix (R)
    """

    user_ids = np.unique(R[:, 0])
    rand_len = int(split_percent * user_ids.shape[0])

    # get random
    selected_u_ids = user_ids[np.random.permutation(user_ids.shape[0])[:rand_len]]

    # get ids
    selected_ids = []
    for selected_u_id in selected_u_ids:
        ids = get_ids_rated_by_x(R, x_id=selected_u_id, x_col_id=0).tolist()
        selected_ids.extend(ids)

    select_i_ids = R[selected_ids,1]
    selected_ratings = R[selected_ids,2]
    selected_u_ids = R[selected_ids,0]

    # make sure everything ok
    assert selected_u_ids.shape[0] == select_i_ids.shape[0]
    assert selected_u_ids.shape[0] == selected_ratings.shape[0]

    return pd.DataFrame(data={'uid':selected_u_ids,'iid':select_i_ids,'rating':selected_ratings},
                        columns=['uid','iid','rating']).as_matrix(), selected_ids

def extract_cold_item_from_R(R, split_percent=0.1):
    """
    :param R: rating matrix
    :param split_percent: indicate number of samples to be extracted
    :return: test cold_item rating matrix (R)
    """

    R_T = R[:,[1,0,2]]
    result, selected_ids = extract_cold_user_from_R(R_T,split_percent)

    return result[:,[1,0,2]], selected_ids

def score_eval_batch(datas,model):
    tf_eval_preds_batch = []

    for i in range(len(datas.eval_batchs)):
        tf_eval_preds, _ = model.run(datas=datas, mode=model.inf_signal, batch_id=i)
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)

    y_nz = [sum(x) > 0 for x in datas.target]
    y_nz = np.arange(datas.target.shape[0])[y_nz]

    preds_all = tf_eval_preds[y_nz, :]
    y = datas.target[y_nz, :]

    x = scipy.sparse.lil_matrix(y.shape)
    x.rows = preds_all
    x.data = np.ones_like(preds_all)
    x = x.toarray()

    z = x * y
    return np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1)))