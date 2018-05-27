import cPickle

import train
import pandas as pd
import argparse

import utils
from nn.isgd import ISGD

parser = argparse.ArgumentParser(description='... PROFILE RECOMMENDATION ...')

parser.add_argument('--train_path',help='define training path',dest='train_path',type=str)
parser.add_argument('--test_path' ,help='define testing path' ,dest='test_path',type=str,default='')
parser.add_argument('--from_save_path' ,help='materials path for loadding old model' ,dest='from_save_path',type=str,default='')
parser.add_argument('--to_save_path' ,help='materials path for storing new model' ,dest='to_save_path',type=str,default='')
parser.add_argument('--k',help='latent dim',dest='k',type=int)
parser.add_argument('--lr',help='learning rate',dest='lr',type=float)
parser.add_argument('--batch_test_size',help='batch size for test',dest='batch_test_size',type=int)
parser.add_argument('--freq_eval',help='number of samples passed to evaluation test set',dest='freq_eval',type=int)
parser.add_argument('--topk',help='get only k items for evaluating',dest='topk',type=int)
parser.add_argument('--n_epochs',help='number of epochs',dest='n_epochs',type=int)

args   = parser.parse_args()

def main(train_path, test_path, from_save_path, to_save_path, k, lr, n_epochs, freq_eval, topk, batch_test_size):
    # load model from save_path
    model, user2id = cPickle.load(open(from_save_path,'r'))
    print ('loaded model from %s ...' % from_save_path)

    # reading training data from file, remember that training file must contain header itself .
    # the same for testing data path.
    train_dataset = pd.read_csv(train_path, header=None)
    assert train_dataset.columns == ['user_name_1', 'user_name_2']
    print ('-- loaded training dataset with %d samples ...' % train_dataset.shape[0])

    if test_path != '':
        test_dataset = pd.read_csv(test_path, header=None)
        assert test_dataset.columns == ['user_name_1', 'user_name_2']

        print ('-- loaded testing dataset with %d samples ...' % test_dataset.shape[0])
    else:
        test_dataset = None

    # create vocabulary (user2id)
    new_user2id = utils.create_vocab(raw_datas=train_dataset.to_records(False,False))
    print ('-- created mapping vocabulary with %d unique names ...' % len(user2id))

    # update old mapping vocabulary with the newer one
    for k,v in new_user2id.items():
        if k not in user2id: user2id[k] = len(user2id)

    train.train(train_dataset=train_dataset,test_dataset=test_dataset,user2id=user2id,model=model,
          batch_test_size=batch_test_size,freq_eval=freq_eval,topk=topk,n_epochs=n_epochs,save_path=to_save_path)

if __name__ == '__main__':
    main(**args)
    pass