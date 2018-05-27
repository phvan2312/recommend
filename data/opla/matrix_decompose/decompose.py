import sys
sys.path.append('..')

import pandas as pd
from scipy.sparse import csr_matrix
from lightfm import LightFM
import cPickle
import argparse

from lightfm.evaluation import recall_at_k, precision_at_k

# rating_path = './../split/train.csv' #'./../split/train.csv'#'./../metadata/rating_matrix.csv'
# save_lightfm_path = './lightfm.pkl'
# save_U_path = './U.csv.bin'
# save_V_path = './V.csv.bin'
# save_U_bias_path = './U_bias.csv.bin'
# save_V_bias_path = './V_bias.csv.bin'

parser  = argparse.ArgumentParser()

parser.add_argument('--rating_path',help='path for rating matrix',dest='rating_path',type=str)
parser.add_argument('--save_lightfm_path',help='path for storing lightfm model class',dest='save_lightfm_path',type=str)
parser.add_argument('--save_U_path',help='path for storing U matrix',dest='save_U_path',type=str)
parser.add_argument('--save_V_path',help='path for storing V matrix',dest='save_V_path',type=str)
parser.add_argument('--save_U_bias_path',help='path for storing U bias',dest='save_U_bias_path',type=str)
parser.add_argument('--save_V_bias_path',help='path for storing V bias',dest='save_V_bias_path',type=str)

args    = vars(parser.parse_args())

"""
USAGE:
python decompose.py --rating_path=./../split/train.csv --save_lightfm_path=./lightfm.pkl --save_U_path=./U.csv.bin
--save_V_path=./V.csv.bin --save_U_bias_path=./U_bias.csv.bin --save_V_bias_path=./V_bias.csv.bin
"""


def main(rating_path,save_lightfm_path,save_U_path,save_V_path,save_U_bias_path,save_V_bias_path):
    rating_df = pd.read_csv(rating_path,sep=',',names=['profile','item','rating'])

    row, col, ratings = rating_df['profile'].values, rating_df['item'].values, rating_df['rating'].values
    n_user = 627 #np.max(row) + 1
    n_item = 12 #np.max(col) + 1

    train_data = csr_matrix((ratings, (row, col)), shape=(n_user, n_item))
    model = LightFM(loss='warp',no_components=200,item_alpha=0.001,user_alpha=0.001)
    model.fit(train_data, epochs=20, num_threads=30)

    print ("u_preference matrix shape: ", model.user_embeddings.shape)
    print ("v_preference matrix shape: ", model.item_embeddings.shape)
    print ("u_bias shape: ", model.user_biases.shape)
    print ("v_bias shape: ", model.item_biases.shape)

    model.user_embeddings.astype('float32').tofile(open(save_U_path, 'w'))
    model.item_embeddings.astype('float32').tofile(open(save_V_path, 'w'))
    model.user_biases.reshape((-1,1)).astype('float32').tofile(open(save_U_bias_path, 'w'))
    model.item_biases.reshape((-1,1)).astype('float32').tofile(open(save_V_bias_path, 'w'))

    cPickle.dump(model, open(save_lightfm_path, 'w'))

def test(train_matrix_path, test_matrix_path):
    train_rating_df = pd.read_csv(train_matrix_path,sep=',',names=['profile','item','rating'])
    train_row, train_col, train_ratings = train_rating_df['profile'].values, train_rating_df['item'].values, \
                                          train_rating_df['rating'].values

    n_user = 627 #np.max(train_row) + 1
    n_item = 12 #np.max(train_col) + 1

    train_data = csr_matrix((train_ratings, (train_row, train_col)), shape=(n_user, n_item))
    model = LightFM(loss='warp', no_components=200, item_alpha=0.001, user_alpha=0.001)
    model.fit(train_data, epochs=20, num_threads=30)

    test_rating_df = pd.read_csv(test_matrix_path,sep=',',names=['profile','item','rating'])
    test_row, test_col, test_ratings = test_rating_df['profile'].values, test_rating_df['item'].values, \
                                       test_rating_df['rating'].values

    test_data = csr_matrix((test_ratings,(test_row, test_col)),shape=(n_user, n_item))
    print("Train precision: %.5f" % recall_at_k(model, train_data, k=6,num_threads=1).mean())
    print("Test precision: %.5f" % recall_at_k(model, test_data, k=6,num_threads=1).mean())

    # print("Train precision: %.5f" % precision_at_k(model, train_data, k=10).mean())
    # print("Test precision: %.5f" % precision_at_k(model, test_data, k=10).mean())

if __name__ == '__main__':
    main(**args)
    #test(train_matrix_path='./../../movielen1m/fake_data/train.csv',test_matrix_path='./../../movielen1m/fake_data/test_warm.csv')
    #test(train_matrix_path='./../split/train.csv',test_matrix_path='./../split/test_warm.csv')

    # rating_path = './../metadata/rating_matrix.csv'
    #
    # rating_datas  = pd.read_csv(rating_path,sep=',')
    # rating_matrix = rating_datas.as_matrix()
    #
    # U, V = matrix_decomposite(rating_matrix, k=200, n_iter=1000)
    #
    # U.astype('float32').tofile(open('U.csv.bin','w'))
    # V.astype('float32').tofile(open('V.csv.bin','w'))