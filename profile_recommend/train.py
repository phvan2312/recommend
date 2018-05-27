import cPickle

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from sklearn.utils import shuffle
import argparse

import utils
from nn.isgd import ISGD

"""
Define some global arguments
"""
parser  = argparse.ArgumentParser(description="ISGD: PROFILE RECOMMENDATION ...")

parser.add_argument('--train_path',help='define training path',dest='train_path',type=str)
parser.add_argument('--test_path' ,help='define testing path' ,dest='test_path',type=str,default='')
parser.add_argument('--save_path' ,help='save materials path' ,dest='save_path',type=str,default='')
parser.add_argument('--k',help='latent dim',dest='k',type=int)
parser.add_argument('--lr',help='learning rate',dest='lr',type=float)
parser.add_argument('--batch_test_size',help='batch size for test',dest='batch_test_size',type=int)
parser.add_argument('--freq_eval',help='number of samples passed to evaluation test set',dest='freq_eval',type=int)
parser.add_argument('--topk',help='get only k items for evaluating',dest='topk',type=int)
parser.add_argument('--n_epochs',help='number of epochs',dest='n_epochs',type=int)

args    = vars(parser.parse_args())

"""
train_path, test_path, save_path, k, lr, n_epochs, batch_test_size = 100, freq_eval = 100, topk = 100
params = {
        'matrix_path' : './twitter/data_mini_train.txt', #matrix_path,
        'test_matrix_path' : './twitter/data_mini_test.txt', #test_matrix_path,
        'user2id_path' : './twitter/data_user2id.txt', #user2id_path,
        'k' : 100,
        'learning_rate' : 0.01,
        'n_epochs': 10
    }

"""

"""
Python usage:
python train.py --train_path=./twitter/data_mini_train.txt --test_path=./twitter/data_mini_test.txt --save_path=./save.pkl
--k=100 --lr=0.001 --batch_test_size=100 --freq_eval=1000 --topk=100 --n_epochs=20
"""


def train(train_dataset, test_dataset, user2id, model, batch_test_size, freq_eval, topk, n_epochs, save_path):
    # convert orginal user_name to their new mapping id
    # do the same for test dataset
    for col in ['user_name_1', 'user_name_2']:
        train_dataset[col] = map(lambda u: user2id[u], train_dataset[col].tolist())

        if test_dataset is not None:
            test_dataset[col] = map(lambda u: user2id[u] if u in user2id else -1, test_dataset[col].tolist())

    # some user names existed in test set does not exist in train set. As a result, it does not exist in
    # mapping vocabulary too. So for this current version, we just remove those samples from test set .
    if test_dataset is not None:
        test_dataset = test_dataset[(test_dataset['user_name_1'] >= 0) & (test_dataset['user_name_2'] >= 0)]
        target_matrix = coo_matrix(
            (
                np.ones(test_dataset.shape[0]),
                (test_dataset['user_name_1'].tolist(), test_dataset['user_name_2'].tolist())
            ),
            shape=(len(user2id) + 1, len(user2id) + 1)
        ).toarray()

    count = 0
    best_test_score = -np.inf

    for n_epoch in range(n_epochs):
        epoch_errs = []
        for i, r in shuffle(train_dataset).iterrows():
            count += 1

            user_1, user_2 = r['user_name_1'], r['user_name_2']
            err = model.update(user_1, user_2)

            epoch_errs += [err]

            if count % freq_eval == 0 and test_dataset is not None and n_epoch > 0:
                # calculate for test set
                n_samples, test_errs = test_dataset.shape[0], []
                """
                for (s, e) in [(s, min(s + batch_test_size, n_samples)) for s in range(0, n_samples, batch_test_size)]:
                    eval_datas = test_dataset.iloc[s:e]
                    eval_datas = eval_datas[eval_datas['user_name_1'].isin(model.known_users)]

                    if eval_datas.shape[0] < 1: continue

                    predict_user_ids = model.recommends(eval_datas['user_name_1'], topk)

                    predict_matrix = coo_matrix(
                        (
                            np.ones(eval_datas.shape[0] * topk),
                            (np.repeat(eval_datas['user_name_1'].tolist(), topk), predict_user_ids.reshape(-1))
                        ),
                        shape=(len(user2id) + 1, len(user2id) + 1)
                    ).toarray()

                    test_err = utils.my_eval(target_matrix, predict_matrix)
                    test_errs += [test_err / float(eval_datas.shape[0])]

                mean_test_err = np.mean(test_errs)
                if mean_test_err > best_test_score:
                    best_test_score = mean_test_err
                    cPickle.dump((model, user2id), open(save_path, 'w'))
                
                """
                eval_datas = test_dataset[test_dataset['user_name_1'].isin(model.known_users)]

                if eval_datas.shape[0] < 1: continue

                predict_user_ids = model.recommends(eval_datas['user_name_1'], topk)

                predict_matrix = coo_matrix(
                    (
                        np.ones(eval_datas.shape[0] * topk),
                        (np.repeat(eval_datas['user_name_1'].tolist(), topk), predict_user_ids.reshape(-1))
                    ),
                    shape=(len(user2id) + 1, len(user2id) + 1)
                ).toarray()

                test_err = utils.my_eval(target_matrix, predict_matrix)

                print ('-- epoch: %d, test error: %.4f ' % (n_epoch + 1, test_err))

        print ('epoch: %d, train error: %.4f ' % (n_epoch + 1, np.mean(epoch_errs)))

    cPickle.dump((model, user2id), open(save_path, 'w'))

def main(train_path, test_path, save_path, k, lr, n_epochs, batch_test_size = 100, freq_eval = 100, topk = 100):
    # reading training data from file, remember that training file must contain header itself .
    # the same for testing data path.
    train_dataset = pd.read_csv(train_path)
    assert train_dataset.columns.tolist() == ['user_name_1','user_name_2']
    print ('-- loaded training dataset with %d samples ...' % train_dataset.shape[0])

    if test_path != '':
        test_dataset  = pd.read_csv(test_path)
        assert test_dataset.columns.tolist() == ['user_name_1', 'user_name_2']

        print ('-- loaded testing dataset with %d samples ...' % test_dataset.shape[0])
    else: test_dataset= None

    # create vocabulary (user2id)
    user2id = utils.create_vocab(raw_datas=train_dataset.to_records(False,False))
    print ('-- created mapping vocabulary with %d unique names ...' % len(user2id))

    model = ISGD(n_user=len(user2id) + 1, n_item=len(user2id) + 1, k=k, learning_rate=lr)

    train(train_dataset=train_dataset,test_dataset=test_dataset,user2id=user2id,model=model,
          batch_test_size=batch_test_size,freq_eval=freq_eval,topk=topk,n_epochs=n_epochs,save_path=save_path)

matrix_path = './fake_data/matrix.txt' # 2 columns (user_name_1, user_name_2), which means user_name_1 likes user_name_2
user2id_path = './fake_data/user2id.txt' # 2 columns (user_name,id)
test_matrix_path = './fake_data/matrix.txt'

if __name__ == '__main__':

    params = {
        'matrix_path' : './twitter/data_mini_train.txt', #matrix_path,
        'test_matrix_path' : './twitter/data_mini_test.txt', #test_matrix_path,
        'user2id_path' : './twitter/data_user2id.txt', #user2id_path,
        'k' : 100,
        'learning_rate' : 0.01,
        'n_epochs': 10
    }

    main(**args)
    pass

