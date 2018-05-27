import numpy as np
import argparse
from scipy.sparse import csc_matrix

# G_ij: i follow j
def pagerank(G, s=0.85, max_err=0.01, max_step = 10000, is_init=False, init_r = None):
    n_samples = G.shape[0] # number of user

    sum_samples = np.sum(G, axis=1, dtype=np.float32) # number of following per user
    norm_G = G[np.where(sum_samples > 0)[0],:] / sum_samples.reshape(-1,1) # nomalize each row by dividing to its sum

    r_0 = np.zeros(n_samples) # array of n_samples zero
    r_1 = np.ones(n_samples) / n_samples if is_init == False else init_r # array of n_samples one - ranking of each user
    assert r_1.shape[0] == n_samples

    n_step = 1

    while np.sum(np.abs(r_1 - r_0)) > max_err and n_step < max_step:
        r_0 = r_1.copy()
        r_1 = s * np.dot(norm_G.T,r_0) + (1-s)*np.ones(n_samples)/n_samples
        n_step += 1

    return r_1

parser  = argparse.ArgumentParser()

parser.add_argument('--matrix_path',help='following matrix path',dest='matrix_path',type=str)
parser.add_argument('--s',help='s',dest='s',type=float)
parser.add_argument('--max_err',help='accpeted error',dest='max_err',type=float)
parser.add_argument('--init_r_path',help='initialize ranking path',dest='init_r_path',type=str,default='./data/user_rank.npy')
parser.add_argument('--out_r_path', help='out ranking path',dest='out_r_path',type=str)

args    = vars(parser.parse_args())

'''
Python Usage:
python pagerank.py --matrix_path=./data/relation_matrix.npy --s=0.85 --max_err=0.000001 --out_r_path=./data/user_rank.npy
'''

def main(matrix_path,s,max_err,init_r_path, out_r_path):
    G = np.load(open(matrix_path,'r'))
    print ('Loaded following matrix from %s with shape: ' % matrix_path, G.shape)

    is_init = False
    init_r  = None

    if init_r_path != '':
        init_r = np.load(open(init_r_path,'r'))
        is_init= True
        print ('Loaded init_ranking vector from %s with shape' % init_r_path, init_r.shape)

    out_r = pagerank(G=G,s=s,max_err=max_err,max_step=10000,is_init=is_init,init_r=init_r)

    # save
    print ('user ranking: ', out_r[:10])
    np.save(open(out_r_path,'w'),out_r)

if __name__ == '__main__':
    #G = cPickle.load()

    # G = np.array([[0, 0, 1, 0, 0, 0, 0],
    #               [0, 1, 1, 0, 0, 0, 0],
    #               [1, 0, 1, 1, 0, 0, 0],
    #               [0, 0, 0, 1, 1, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 1],
    #               [0, 0, 0, 0, 0, 1, 1],
    #               [0, 0, 0, 1, 1, 0, 1]])
    #
    # np.save(open('./data/relation_matrix.npy','w'),G)
    #cPickle.dump(G,open('./data/relation_matrix.pkl','w'))

    #print pagerank(G,s=0.86,max_err=.0001)

    main(**args)

    pass