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

    n_user = 1000 #np.max(rating_datas['profile'].values) + 1
    n_item = 1000 #np.max(rating_datas['item'].values) + 1

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
    def __init__(self, n_user, n_item, k, l2_reg=0.01, learning_rate=0.005):
        self.k = k
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.known_users = np.array([])
        self.known_items = np.array([])
        self.n_user = n_user
        self.n_item = n_item
        self.U = self.__init_W(n_user,self.k) #np.random.normal(0., 0.1, (n_user, self.k))
        self.V = self.__init_W(n_item,self.k) #np.random.normal(0., 0.1, (n_item, self.k))

    def __init_W(self, dim_1, dim_2):
        return np.random.normal(0.,.1, (dim_1,dim_2))

    def update(self, u_index, i_index):

        if u_index not in self.known_users:
            self.known_users = np.append(self.known_users, u_index)
            if u_index > self.U.shape[0]:
                margin = u_index + 1 - self.U.shape[0]
                self.U = np.concatenate([self.U,self.__init_W(margin,self.k)],axis=0)

        u_vec = self.U[u_index]

        if i_index not in self.known_items:
            self.known_items = np.append(self.known_items, i_index)
            if i_index > self.V.shape[0]:
                margin = i_index + 1 - self.V.shape[0]
                self.V = np.concatenate([self.V,self.__init_W(margin,self.k)],axis=0)

        i_vec = self.V[i_index]

        err = 1. - np.inner(u_vec, i_vec)
        if err is np.nan:
            raise Exception('err')

        self.U[u_index] = u_vec + self.learning_rate * (err * i_vec - self.l2_reg * u_vec)
        self.V[i_index] = i_vec + self.learning_rate * (err * u_vec - self.l2_reg * i_vec)

        return err

    def recommends(self, u_indexs, N):
        """
        Recommend Top-N items for list of users
        """
        for u_index in u_indexs:
            if u_index not in self.known_users: raise ValueError('Error: the user is not known.')

        recos  = []
        scores = np.abs(1. - np.dot(np.array([self.U[u_indexs]]), self.V.T)).reshape((len(u_indexs),self.V.shape[0]))

        return np.argsort(scores,axis=1)[:,:N]

    def recommend(self, u_index, N):
        """
        Recommend Top-N items for the user u
        """

        if u_index not in self.known_users: raise ValueError('Error: the user is not known.')

        recos = []
        scores = np.abs(1. - np.dot(np.array([self.U[u_index]]), self.V.T)).reshape(self.V.shape[0])

        return np.argsort(scores)[:N]


if __name__ == '__main__':
    main()

    rating_datas = pd.read_csv(trained_rating_path, sep=',')
    update(rating_datas)

    pass