import sys
sys.path.append('..')

import numpy as np

import pandas as pd
import cPickle

U_path = './U.csv.bin'
V_path = './V.csv.bin'
isgd_path = './saved_model/isgd.pkl'
trained_rating_path = './trained_rating_matrix.csv'

def main():
    rating_path = './../metadata/rating_matrix.csv'
    rating_datas = pd.read_csv(rating_path, sep=',')

    n_user = np.max(rating_datas['profile'].values) + 1
    n_item = np.max(rating_datas['item'].values) + 1

    isgd = train(ISGD(n_user,n_item,k=100),rating_datas)

    # store U,V
    save(isgd.U,isgd.V,rating_datas,isgd)

def train(isgd,ratings):
    for i,r in ratings.iterrows():
        u_id, i_id, rating = r['profile'], r['item'], r['rating']
        isgd.update(u_id,i_id,rating)

    return  isgd

def save(U,V,ratings,isgd):
    U.astype('float32').tofile(open(U_path, 'w'))
    V.astype('float32').tofile(open(V_path, 'w'))
    ratings.to_csv(trained_rating_path,sep=',')
    cPickle.dump(isgd,open(isgd_path,'w'))

def update(ratings_v2):
    ratings_v1 = pd.read_csv(trained_rating_path, sep=',')
    total, diff = get_new_ratings(ratings_v1,ratings_v2)

    isgd = train(cPickle.load(open(isgd_path,'r')), diff)

    # store U,V
    save(isgd.U,isgd.V,total,isgd)

def get_new_ratings(rating_v1, rating_v2):
    # compare rating_v1 and rating_v2, then ouput new_rating = raitng_v2 - rating_v1
    # Mean: return only element rating_v2 has but rating_v1 does not
    # rating_v1 and rating_v2 is DataFrame type

    total = rating_v1.append(rating_v2).drop_duplicates()
    diff = total.append(rating_v1).drop_duplicates(keep=False)

    return total, diff

class ISGD:
    def __init__(self, n_user, n_item, k, l2_reg=0.01, learn_rate=0.005):
        self.k = k
        self.l2_reg = l2_reg
        self.learn_rate = learn_rate
        self.known_users = np.array([])
        self.known_items = np.array([])
        self.n_user = n_user
        self.n_item = n_item
        self.U = np.random.normal(0., 0.1, (n_user, self.k))
        self.V = np.random.normal(0., 0.1, (n_item, self.k))

    def update(self, u_index, i_index,rating = 1.):

        if u_index not in self.known_users: self.known_users = np.append(self.known_users, u_index)
        u_vec = self.U[u_index]

        if i_index not in self.known_items: self.known_items = np.append(self.known_items, i_index)
        i_vec = self.V[i_index]

        err = rating - np.inner(u_vec, i_vec)
        if err is np.nan:
            raise Exception('err')

        self.U[u_index] = u_vec + self.learn_rate * (err * i_vec - self.l2_reg * u_vec)
        self.V[i_index] = i_vec + self.learn_rate * (err * u_vec - self.l2_reg * i_vec)

    def recommend(self, u_index, N, history_vec):
        """
        Recommend Top-N items for the user u
        """

        if u_index not in self.known_users: raise ValueError('Error: the user is not known.')

        recos = []
        scores = np.abs(1. - np.dot(np.array([self.U[u_index]]), self.V.T)).reshape(self.V.shape[0])

        cnt = 0
        for i_index in np.argsort(scores):
            if history_vec[i_index] == 1: continue
            recos.append(i_index)
            cnt += 1
            if cnt == N: break

        return recos

if __name__ == '__main__':
    main()

    rating_datas = pd.read_csv(trained_rating_path, sep=',')
    update(rating_datas)

    pass