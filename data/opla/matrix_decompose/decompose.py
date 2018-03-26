import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from lightfm import LightFM
import cPickle

from lightfm.evaluation import recall_at_k

rating_path = './../split/train.csv'#'./../metadata/rating_matrix.csv'
lightfm_path = './lightfm.pkl'
U_path = './U.csv.bin'
V_path = './V.csv.bin'
U_bias_path = './U_bias.csv.bin'
V_bias_path = './V_bias.csv.bin'

def main():
    rating_df = pd.read_csv(rating_path,sep=',',names=['profile','item','rating'])

    row, col, ratings = rating_df['profile'].values, rating_df['item'].values, rating_df['rating'].values
    n_user = np.max(row) + 1
    n_item = np.max(col) + 1

    train_data = csr_matrix((ratings, (row, col)), shape=(n_user, n_item))
    model = LightFM(loss='warp',no_components=200,item_alpha=0.001,user_alpha=0.001)
    model.fit(train_data, epochs=200, num_threads=2)

    model.user_embeddings.astype('float32').tofile(open(U_path, 'w'))
    model.item_embeddings.astype('float32').tofile(open(V_path, 'w'))
    model.user_biases.reshape((-1,1)).astype('float32').tofile(open(U_bias_path, 'w'))
    model.item_biases.reshape((-1,1)).astype('float32').tofile(open(V_bias_path, 'w'))

def test(train_matrix_path, test_matrix_path):
    train_rating_df = pd.read_csv(train_matrix_path,sep=',',names=['profile','item','rating'])
    train_row, train_col, train_ratings = train_rating_df['profile'].values, train_rating_df['item'].values, \
                                          train_rating_df['rating'].values

    n_user = np.max(train_row) + 1
    n_item = np.max(train_col) + 1

    train_data = csr_matrix((train_ratings, (train_row, train_col)), shape=(n_user, n_item))
    model = LightFM(loss='warp', no_components=200, item_alpha=0.001, user_alpha=0.001)
    model.fit(train_data, epochs=200, num_threads=1)

    test_rating_df = pd.read_csv(test_matrix_path,sep=',',names=['profile','item','rating'])
    test_row, test_col, test_ratings = test_rating_df['profile'].values, test_rating_df['item'].values, \
                                       test_rating_df['rating'].values

    test_data = csr_matrix((test_ratings,(test_row, test_col)),shape=(n_user, n_item))
    print("Train precision: %.5f" % recall_at_k(model, train_data, k=5).mean())
    print("Test precision: %.5f" % recall_at_k(model, test_data, k=5).mean())

if __name__ == '__main__':
    #main()
    test(train_matrix_path='./../split/train.csv',test_matrix_path='./../split/test_warm.csv')

    # rating_path = './../metadata/rating_matrix.csv'
    #
    # rating_datas  = pd.read_csv(rating_path,sep=',')
    # rating_matrix = rating_datas.as_matrix()
    #
    # U, V = matrix_decomposite(rating_matrix, k=200, n_iter=1000)
    #
    # U.astype('float32').tofile(open('U.csv.bin','w'))
    # V.astype('float32').tofile(open('V.csv.bin','w'))