import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
#from utils import extract_cold_item_from_R, extract_cold_user_from_R, extract_warm_from_R, matrix_decomposite

def build_one_hot(datas):
    """
    :param datas: type list, list of features for each user/item
    :return: OnehotEncoder, preprocessed data
    """
    enc = OneHotEncoder()
    enc.fit(datas)
    preprocess_data = enc.transform(datas)

    return enc, preprocess_data

def build_tfidf(datas):
    """
    :param datas: type list, list of features for each user/item
    :return: tfidf, preprocessed data
    """
    def my_preprocessing(raw_str):
        return raw_str.lower()

    def my_tokenization(raw_str):
        return raw_str.split('|')

    tfidf_vect = TfidfVectorizer(preprocessor=my_preprocessing,tokenizer=my_tokenization,max_features=200)
    preprocess_data = tfidf_vect.fit_transform(datas)

    return tfidf_vect, preprocess_data

def dump_R(R, data_path):
    u_ids, i_ids, rating = R.T.tolist()
    df = pd.DataFrame({'uid':u_ids,'iid':i_ids,'rating':rating},columns=['uid','iid','rating'],dtype='int32')
    df.to_csv(data_path,sep=',',header=False,index=False)

def dump_libsvm(vectors, path):
    len = vectors.shape[0]
    targets = np.array(range(len))

    dump_svmlight_file(X=vectors,y=targets,f=path)

def dump_numpy(data,path):
    data.tofile(open(path,'w'))

def load_data(data_path,names,sep='::'):
    df = pd.read_csv(data_path,sep=sep,names=names)
    return df

def remove_R_by_ids(R,ids):
    org_n_rows = R.shape[0]
    remain_row_ids = set(range(org_n_rows)) - set(ids)

    return R[list(remain_row_ids),:]

def dump_ids(data, path):
    with open(path,'w') as f:
        f.write('\n'.join(data))

def get_x_ids(R,x_col_id):
    ids = list(set(R[:,x_col_id]))
    ids_str = [str(id) for id in ids]
    return ids_str

if __name__ == '__main__':
    rating_path = './ratings.dat'
    movie_path  = './movies.dat'
    user_path   = './users.dat'

    # for rating matrix
    rating_datas = load_data(data_path=rating_path,names=['uid','iid','rating','timestamp'],sep='::')
    rating_datas['uid'] = rating_datas['uid'].astype('int32') - 1 # convert user_based from 1 to 0
    rating_datas['iid'] = rating_datas['iid'].astype('int32') - 1 # convert item_based from 1 to 0
    del rating_datas['timestamp'] # no need

    # for item content
    item_datas = load_data(data_path=movie_path,names=['iid','title','category'],sep='::')
    item_datas['iid'] = item_datas['iid'].astype('int32') - 1
    del item_datas['title']

    # for user content
    user_datas = load_data(data_path=user_path,names=['uid','gender','age','occupation','nonsense'],sep='::')
    user_datas['uid'] = user_datas['uid'].astype('int32') - 1
    user_datas['gender'] = map(lambda x: 1 if x == 'F' else 0 , user_datas['gender'])
    del user_datas['nonsense']

    # get vector
    train_item_datas = item_datas['category'].tolist()
    train_user_datas = user_datas[['gender','age','occupation']].values.tolist()

    _, item_content_vectors = build_tfidf(train_item_datas)
    full_item_content_vectors = np.zeros(shape=(3952,item_content_vectors.shape[1]),dtype=np.float32)
    for iid, item_content_vector in zip(item_datas['iid'], item_content_vectors.toarray()):
        full_item_content_vectors[iid,:] = item_content_vector
    item_content_vectors = full_item_content_vectors

    _, user_content_vectors = build_one_hot(train_user_datas)

    # U,V content matrix
    dump_libsvm(vectors=user_content_vectors, path='./split_data/user_features_0based.txt')
    dump_libsvm(vectors=item_content_vectors, path='./split_data/item_features_0based.txt')

    # U,V preference matrix
    rating_matrix = rating_datas.as_matrix()
    U,V = matrix_decomposite(rating_matrix,k=200,n_iter=100)

    dump_numpy(data=U.astype('float32'), path='./split_data/trained/warm/U.csv.bin')
    dump_numpy(data=V.astype('float32'), path='./split_data/trained/warm/V.csv.bin')

    # For cold_user
    cold_user_matrix, selected_ids = extract_cold_user_from_R(rating_matrix, 0.1)
    dump_R(R=cold_user_matrix, data_path='./split_data/warm/test_cold_user.csv')
    dump_ids(data=get_x_ids(cold_user_matrix, 1), path='./split_data/warm/test_cold_user_item_ids.csv')
    rating_matrix = remove_R_by_ids(rating_matrix, selected_ids)

    # For cold item
    cold_item_matrix, selected_ids = extract_cold_item_from_R(rating_matrix, 0.1)
    dump_R(R=cold_item_matrix, data_path='./split_data/warm/test_cold_item.csv')
    dump_ids(data=get_x_ids(cold_item_matrix, 1), path='./split_data/warm/test_cold_item_item_ids.csv')
    rating_matrix = remove_R_by_ids(rating_matrix, selected_ids)

    # For warm user
    warm_matrix, selected_ids = extract_warm_from_R(rating_matrix, 0.2)
    dump_R(R=warm_matrix, data_path='./split_data/warm/test_warm.csv')
    dump_ids(data=get_x_ids(warm_matrix, 1), path='./split_data/warm/test_warm_item_ids.csv')
    rating_matrix = remove_R_by_ids(rating_matrix, selected_ids)

    # For train.csv
    dump_R(R=rating_matrix, data_path='./split_data/warm/train.csv')