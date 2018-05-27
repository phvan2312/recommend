import tensorflow as tf
import numpy as np
import cPickle
from nn.nn_with_dropoutnet import RecommendNet
import os
from lightfm import LightFM
from data.opla import opla_utils
from data.opla.vect.deep.vectorizer import VectorizerModel, extract_work_from_user, extract_skill_from_user

import pandas as pd
import json
from scipy.sparse import coo_matrix

n_items = 12

class Vectorizer:
    def __init__(self, class_path,deep_path):
        self.vect = cPickle.load(open(class_path, 'r'))
        with tf.Graph().as_default():
            self.vect.build()
            self.vect.load(deep_path)

    def get_user_features(self, user_profiles):
        """
        :param user_profiles: list of json
        :return:
        """
        profile_works = [extract_work_from_user(user_profile['work']) for user_profile in user_profiles.values()]
        profile_skills = [extract_skill_from_user(user_profile['skills']) for user_profile in user_profiles.values()]

        assert len(profile_works) == len(profile_skills)
        profile_vects = self.vect .get_profile_vector(works=profile_works, skills=profile_skills)

        return profile_vects


    def get_item_features(self, item_profiles):
        """

        :param item_profiles: dont care
        :return:
        """
        return self.vect.get_item_vector(item_id=range(n_items))

class OldModel:
    def __init__(self, model_path, model_deep_path, profilevocab_path, itemvocab_path, vectorizer_class_path,
                 vectorizer_deep_path, **kargs):
        assert os.path.exists(model_path)
        #assert os.path.exists(model_deep_path)

        # restore model
        print ('-- restore model')

        self.model = cPickle.load(open(model_path,'r'))
        self.model.build()
        self.model.restore(model_deep_path)

        # restore vocab
        print ('-- restore vocabulary')

        profile_df = pd.read_csv(profilevocab_path, sep=',', encoding='utf-8')
        item_df = pd.read_csv(itemvocab_path, sep=',', encoding='utf-8')
        self.profile2id = {v:k for k,v in zip(profile_df['id'].tolist(),profile_df['org_name'].tolist())}
        self.item2id = {v:k for k,v in zip(item_df['id'].tolist(),item_df['org_name'].tolist())}

        self.vectorizer = Vectorizer(vectorizer_class_path,vectorizer_deep_path)

    def load_lightfm_matrix(self, lightfm_path):
        assert os.path.exists(lightfm_path)
        lightfm = cPickle.load(open(lightfm_path, 'r'))

        u_pref_matrix = lightfm.user_embeddings.astype('float32')
        self.u_pref_zero_matrix = np.vstack([u_pref_matrix, np.zeros_like(u_pref_matrix[0, :])])

        v_pref_matrix = lightfm.item_embeddings.astype('float32')
        self.v_pref_zero_matrix = np.vstack([v_pref_matrix, np.zeros_like(v_pref_matrix[0, :])])

        u_bias_matrix = lightfm.user_biases.reshape((-1,1)).astype('float32')
        v_bias_matrix = lightfm.item_biases.reshape((-1,1)).astype('float32')

        self.u_bias_zero_matrix = np.vstack([u_bias_matrix, np.zeros_like(u_bias_matrix[0,:])])
        self.v_bias_zero_matrix = np.vstack([v_bias_matrix, np.zeros_like(v_bias_matrix[0,:])])

    def predict(self, user_profiles):
        """
        :param user_profiles: list of json
        :return:
        """

        datas = {k:v for k,v in enumerate(user_profiles)}
        utility_matrix, item_details, profile_details = opla_utils.get_metadata(datas=datas)

        u_cont = self.vectorizer.get_user_features(profile_details)
        v_cont = self.vectorizer.get_item_features(item_details)

        u_ids  = [self.profile2id[u] if u in self.profile2id else -1 for u in profile_details.keys()]

        u_pref = self.u_pref_zero_matrix[u_ids]
        u_bias = self.u_bias_zero_matrix[u_ids]
        v_pref = self.v_pref_zero_matrix[:-1,:]
        v_bias = self.v_bias_zero_matrix[:-1,:]

        ip_feed_dict = {
            self.model.u_pref: u_pref,
            self.model.v_pref: v_pref,
            self.model.u_bias: u_bias,
            self.model.v_bias: v_bias,
            self.model.u_content: u_cont,
            self.model.v_content: v_cont,
            self.model.phase: 0
        }

        _,topk_col_ids = self.model.sess.run(tf.nn.top_k(self.model.predicted_R, k=n_items, sorted=True),
                            ip_feed_dict)
        return topk_col_ids

def my_eval_metric(target_matrix, predict_matrix):
    target_nonzezo = [sum(x) > 0 for x in target_matrix]
    target_nonzero = np.arange(0,len(target_matrix))[target_nonzezo]

    processed_target_matrix = target_matrix[target_nonzero]
    processed_predict_matrix = predict_matrix[target_nonzero]

    mul = processed_target_matrix * processed_predict_matrix

    return np.mean(np.sum(mul,axis=1) / np.sum(processed_target_matrix, axis=1))

import argparse

parser  = argparse.ArgumentParser()

parser.add_argument('--model_path',type=str,dest='model_path',help='path storing model class')
parser.add_argument('--model_deep_path',type=str,dest='model_deep_path',help='path storing weight of model')
parser.add_argument('--profilevocab_path',type=str,dest='profilevocab_path',help='id2profile dataframe path')
parser.add_argument('--itemvocab_path',type=str,dest='itemvocab_path',help='id2item dataframe path')
parser.add_argument('--vectorizer_class_path',type=str,dest='vectorizer_class_path',help='vectorizer_class_path')
parser.add_argument('--vectorizer_deep_path',type=str,dest='vectorizer_deep_path',help='vectorizer_deep_path')
parser.add_argument('--lightfm_path',type=str,dest='lightfm_path',help='lightfm_path')

args    = vars(parser.parse_known_args()[0])

"""
USAGE:
python eval_v2.py --model_path=./saved_model/dropoutnet_opla_vTest/model.pkl --model_deep_path=./saved_model/dropoutnet_opla_vTest/model.ckpt
--profilevocab_path=./data/opla/metadata/out/id2profile.csv --itemvocab_path=./data/opla/metadata/out/id2item.csv --vectorizer_class_path=./data/opla/vect/deep/saved_model/model.pkl --vectorizer_deep_path=./data/opla/vect/deep/saved_model/deepvec.ckpt --lightfm_path=./data/opla/matrix_decompose/lightfm.pkl
"""

def get_json_from_dir(dir_path):
    json_datas = []

    for path in os.listdir(dir_path):
        json_data = json.load(path)
        json_datas.append(json_data)

    return json_datas

def eval(json_datas,model=None,n_user=629,n_item=12,topk=5):
    row_ids = []
    col_ids = []

    # get target matrix
    for i, json_data in enumerate(json_datas):
        cats = json_data['category'].keys()  # category

        cats_ids = [old.item2id[cat] for cat in cats]
        prof_ids = [i] * len(cats)

        row_ids = row_ids + prof_ids
        col_ids = col_ids + cats_ids

    # target
    target_matrix = coo_matrix(
        (
            np.ones(len(row_ids)),
            (row_ids, col_ids)
        ),
        shape=(n_user, n_item)
    ).toarray()

    # get predict
    predict_col_ids = old.predict(json_datas)
    predict_col_ids = predict_col_ids[:, :topk]

    row_ids = np.repeat(range(len(json_datas)), topk)
    col_ids = predict_col_ids.reshape(-1)

    predict_matrix = coo_matrix(
        (
            np.ones(len(row_ids)),
            (row_ids, col_ids)
        ),
        shape=(n_user, n_item)
    ).toarray()

    return my_eval_metric(target_matrix, predict_matrix)

if __name__ == '__main__':

    # test
    old_params = {
        'model_path': './saved_model/vTest/model.pkl', # model_class_path ()
        'model_deep_path': './saved_model/vTest/model.ckpt', # load values of matrix tensorflow
        'profilevocab_path': './fake_data/opla/metadata/id2profile.csv', # id_profile df
        'itemvocab_path': './fake_data/opla/metadata/id2item.csv', # id_item df
        'vectorizer_class_path': './fake_data/opla/vect/deep/saved_model/model.pkl', # vectorizer_model_class
        'vectorizer_deep_path': './fake_data/opla/vect/deep/saved_model/deepvec.ckpt' # load values of matrix tensorflow
    }

    # load old model
    old = OldModel(**args)
    old.load_lightfm_matrix(lightfm_path=args['lightfm_path'])

    #exit()

    json_data = json.load(open('./data/opla/raw/zipper-2018-01-23--08-01/acyras.json'))

    json_datas = [json_data]

    print eval(json_datas,old,629,12,5)
