import tensorflow as tf
import pandas as pd
from nn.nn_with_dropoutnet import RecommendNet
from utils import load_ndarray_data
import cPickle
import json

from data.opla.opla_utils import get_metadata
from data.opla.vect.profile_vectorizer import ProfileVectorizer

class EvalRecommendNet:
    def __init__(self, params_path, profile_vocab_path, item_vocab_path, **args):
        self.params = self.__load_params(params_path)
        self.model  = self.__load_model(self.params['tf_model_path'],self.params['tf_model_params'])

        self.profile2id, self.id2profile = self.__load_vocab(profile_vocab_path)
        self.item2id, self.id2item = self.__load_vocab(item_vocab_path)

        latent_dim = self.params['tf_model_params']['latent_dim']
        user_feature_dim = self.params['tf_model_params']['user_feature_dim']
        item_feature_dim = self.params['tf_model_params']['item_feature_dim']
        n_user = self.params['n_user'] # include zero_user_id (last index)
        n_item = self.params['n_item'] # include zero_item_id (last index)

        self.zero_user_id = n_user
        self.zero_item_id = n_item

        self.u_pref = load_ndarray_data(self.params['u_pref_path'],'bin').reshape(n_user, latent_dim)
        self.v_pref = load_ndarray_data(self.params['v_pref_path'],'bin').reshape(n_item, latent_dim)
        self.u_cont = load_ndarray_data(self.params['u_cont_path'],'bin').reshape(n_user - 1, user_feature_dim)
        self.v_cont = load_ndarray_data(self.params['v_cont_path'],'bin').reshape(n_item - 1, item_feature_dim)

        self.model_topk = self.params['tf_model_params']['k']

    def __load_vocab(self, path):
        vocab = pd.read_csv(path,sep=',',encoding='utf-8')

        x2id = dict((e, i) for i, e in zip(vocab.id, vocab.org_name))
        id2x = dict((i, e) for i, e in zip(vocab.id, vocab.org_name))

        return x2id, id2x

    def __load_params(self, params_path):
        params = cPickle.load(open(params_path,'r'))
        return params

    def __load_model(self, model_path, model_params):
        with tf.device('/cpu:0'):
            model = RecommendNet(**model_params)
            model.build(build_session=True)

            model.restore(model_path)

        return model

    def predict_with_matrix(self, u_ids, u_cont, topk = 10):
        assert topk <= self.model_topk

        # model need 4 inputs: u_pref, v_pref, u_cont, v_cont
        feed_dict = {
            self.model.u_pref: self.u_pref[u_ids, :],
            self.model.v_pref: self.v_pref[:-1, :],  # just remove the last row
            self.model.u_content: u_cont,
            self.model.v_content: self.v_cont,
            self.model.phase: 0
        }

        topk_col_ids = self.model.sess.run(self.model.predicted_topk, feed_dict)
        return topk_col_ids[:, :topk]


    def predict(self, uids, topk = 10):
        assert topk <= self.model_topk

        # model need 4 inputs: u_pref, v_pref, u_cont, v_cont
        feed_dict = {
            self.model.u_pref : self.u_pref[uids,:],
            self.model.v_pref : self.v_pref[:-1,:], # just remove the last row
            self.model.u_content : self.u_cont[uids,:],
            self.model.v_content : self.v_cont,
            self.model.phase: 0
        }

        topk_col_ids = self.model.sess.run(self.model.predicted_topk,feed_dict)
        return topk_col_ids[:,:topk]

### Show time execution ###
import time
time_log = {}
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            time_log[method.__name__] = '%2.2f ms' % ((te - ts) * 1000)
            print '%r : %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result

    return timed
### ###

class Test:
    def __init__(self, profile_vectorizer_path, profile_vocab_path):
        self.profile_vectorizer = cPickle.load(open(profile_vectorizer_path,'r'))
        self.profile2id, self.id2profile = self.__load_vocab(profile_vocab_path)

    def __load_vocab(self, path):
        vocab = pd.read_csv(path,sep=',',encoding='utf-8')

        x2id = dict((e, i) for i, e in zip(vocab.id, vocab.org_name))
        id2x = dict((i, e) for i, e in zip(vocab.id, vocab.org_name))

        return x2id, id2x

    @timeit
    def convert_from_json(self, lst_json):
        data = {i:json_dict for i,json_dict in enumerate(lst_json)}

        _, _, profile_details = get_metadata(datas=data)

        i = 1

        input = {
            'education' : [v['education'] for _,v in profile_details.items()] * i,
            'skills' : [v['skills'] for _,v in profile_details.items()] * i,
            'work' : [v['work'] for _,v in profile_details.items()] * i
        }

        u_cont_matrix = self.profile_vectorizer.predict(input)
        u_ids = [self.profile2id[u_name] if u_name in self.profile2id.keys() else len(self.profile2id)
                 for u_name in profile_details.keys()]
        u_names = profile_details.keys()

        return u_cont_matrix, u_ids, u_names

if __name__ == '__main__':
    params = {
        'params_path' : './saved_model/dropoutnet_opla_v1/params.pkl',
        'profile_vocab_path': './data/opla/metadata/id2profile.csv',
        'item_vocab_path': './data/opla/metadata/id2item.csv'
    }

    eval_Rec = EvalRecommendNet(**params)

    test = Test(profile_vectorizer_path = './data/opla/vect/profile_vectorizer.pkl',
                profile_vocab_path = './data/opla/metadata/id2profile.csv')

    # test with one json file

    json_data = json.load(open('./data/opla/raw/zipper-2018-01-23--08-01/acyras.json'))
    json_data

    u_cont_matrix, u_ids, u_names = test.convert_from_json(lst_json=[json_data])

    topk_iids = eval_Rec.predict_with_matrix(u_ids = u_ids, u_cont = u_cont_matrix, topk=3)

    for u_name, top_iid in zip(u_names,topk_iids):
        recommend_items = '; '.join([eval_Rec.id2item[iid] for iid in top_iid])
        print "Recommend items: %s for user: %s \n" % (recommend_items, u_name)

    pass