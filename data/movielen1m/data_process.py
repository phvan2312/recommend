import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from recommend.utils import extract_cold_user_from_R, extract_warm_from_R

def remove_R_by_ids(R,ids):
    org_n_rows = R.shape[0]
    remain_row_ids = set(range(org_n_rows)) - set(ids)

    return R[list(remain_row_ids),:]

ratings_path  = './ratings.dat'
user_profiles_path = './users.dat'
item_profiles_path = './movies.dat'

n_users = 6040
n_items = 3952

user_feature_path = './data/user_feature.csv.bin'
item_feature_path = './data/item_feature.csv.bin'
cold_user_path = './data/test_cold_user.csv'
warm_path = './data/test_warm.csv'
train_path = './data/train.csv'

if __name__ == '__main__':
    ratings_df = pd.read_csv(ratings_path,sep='::',names=['uid','iid','rating','timestamp'])
    del ratings_df['timestamp']

    # user_infos = pd.read_csv(user_profiles_path,sep='::',names=['uid','gender','age','occupation','zip_code'])
    # del user_infos['zip_code']
    #
    # item_infos = pd.read_csv(item_profiles_path,sep='::',names=['iid','title','genre'])
    # del item_infos['title']
    #
    # print ('-- processing item profile')
    # processed_item_infos = pd.DataFrame(columns=['iid','genre'])
    #
    # for item_id, item_row in item_infos.iterrows():
    #     if item_row['genre'].find('|') > 0:
    #
    #         for genre in item_row['genre'].split('|'):
    #             processed_item_infos = processed_item_infos.append(
    #                 {
    #                     'iid' : int(item_row['iid']),
    #                     'genre' : genre
    #                 }
    #                 , ignore_index=True
    #             )
    #     else:
    #         processed_item_infos = processed_item_infos.append(
    #             {
    #                 'iid': int(item_row['iid']),
    #                 'genre': item_row['genre']
    #             }
    #             , ignore_index=True
    #         )
    #
    # del item_infos
    # item_infos = processed_item_infos
    #
    # user_label_encoders = {
    #     'gender' : LabelEncoder(),
    #     'age' : LabelEncoder(),
    #     'occupation' : LabelEncoder(),
    # }
    #
    # item_label_encoders = {
    #     'genre' : LabelEncoder()
    # }
    #
    #
    # # label encoder
    # print ('-- build label encoding for converting interger/string... to integer')
    #
    # for col_name in ['gender','age','occupation']: user_label_encoders[col_name].fit(user_infos[col_name].tolist())
    # for col_name in ['genre']: item_label_encoders[col_name].fit(item_infos[col_name].tolist())
    #
    # for col_name in ['gender','age','occupation']:
    #     user_infos[col_name] = user_label_encoders[col_name].transform(user_infos[col_name].tolist())
    #
    # for col_name in ['genre']:
    #     item_infos[col_name] = item_label_encoders[col_name].transform(item_infos[col_name].tolist())
    #
    # # one hot encoder
    # print ('-- building onehot encoding ')
    #
    # user_onehot_encoder = OneHotEncoder(sparse=False,dtype='float32')
    # item_onehot_encoder = OneHotEncoder(sparse=False,dtype='float32')
    #
    # user_infos['onehot_encode'] = list(user_onehot_encoder.fit_transform(user_infos[['gender','age','occupation']].as_matrix()))
    # item_infos['onehot_encode'] = list(item_onehot_encoder.fit_transform(item_infos[['genre']].as_matrix()))
    #
    # print ('-- building profile/item feature matrix ')
    #
    # user_feature_matrix = np.zeros(shape=(n_users,len(user_infos['onehot_encode'][0])),dtype='float32')
    # item_feature_matrix = np.zeros(shape=(n_items,len(item_infos['onehot_encode'][0])),dtype='float32')
    #
    # uid_vector_df = user_infos[['uid','onehot_encode']]
    # iid_vector_df = item_infos[['iid','onehot_encode']].groupby(['iid'],as_index=False).sum()
    #
    # user_feature_matrix[uid_vector_df['uid'].values.astype('int32') - 1,:] = uid_vector_df['onehot_encode'].tolist()
    # item_feature_matrix[iid_vector_df['iid'].values.astype('int32') - 1,:] = iid_vector_df['onehot_encode'].tolist()
    #
    # user_feature_matrix.astype('float32').tofile(open(user_feature_path,'w'))
    # item_feature_matrix.astype('float32').tofile(open(item_feature_path,'w'))
    #
    # print ('-- split rating matrix into 3 smaller matrixs: test_cold_user, test_warm, train')

    rating_matrix = ratings_df.as_matrix()

    # cold_user matrix
    cold_user_matrix, selected_ids = extract_cold_user_from_R(rating_matrix, 0.1)
    rating_matrix = remove_R_by_ids(rating_matrix, selected_ids) # get remain
    pd.DataFrame({'profile':cold_user_matrix[:,0] - 1,'item':cold_user_matrix[:,1] - 1,'rating':cold_user_matrix[:,2]}).\
        to_csv(cold_user_path,sep=',',header=False,index=False, columns=['profile','item','rating'])

    # warm matrix
    warm_matrix, selected_ids = extract_warm_from_R(rating_matrix, 0.2)
    rating_matrix = remove_R_by_ids(rating_matrix, selected_ids)  # get remain
    pd.DataFrame({'profile': warm_matrix[:, 0] - 1, 'item': warm_matrix[:, 1] - 1, 'rating': warm_matrix[:, 2]}). \
        to_csv(warm_path, sep=',', header=False, index=False, columns=['profile','item','rating'])

    # train matrix
    pd.DataFrame({'profile': rating_matrix[:, 0] - 1, 'item': rating_matrix[:, 1] - 1, 'rating': rating_matrix[:, 2]}). \
        to_csv(train_path, sep=',', header=False, index=False, columns=['profile','item','rating'])

    pass