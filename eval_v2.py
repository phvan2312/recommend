import tensorflow as tf
import numpy as np
import cPickle
from nn.nn_with_dropoutnet import RecommendNet
import os
from lightfm import LightFM
from data.opla import opla_utils
from data.opla.vect.deep.vectorizer import ItemVectorizerModel, extract_work_from_user, extract_skill_from_user

n_items = 12

class Vectorizer:
    def __init__(self, saved_path):
        self.vect = cPickle.load(open(saved_path,'w'))

    def get_user_features(self, user_profiles):
        """
        :param user_profiles: list of json
        :return:
        """
        profile_works = [extract_work_from_user(user_profile['work']) for user_profile in user_profiles]
        profile_skills = [extract_skill_from_user(user_profile['skills']) for user_profile in user_profiles]

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
    def __init__(self, model_path, model_deep_path, profile2id_path, item2id_path):
        assert os.path.exists(model_path)
        assert os.path.exists(model_deep_path)

        # restore model
        print ('-- restore model')

        self.model = cPickle.load(open(model_path,'w'))
        self.model.build()
        self.model.restore(model_deep_path)

        # restore vocab
        print ('-- restore vocabulary')
        self.profile2id = cPickle.load(open(profile2id_path,'w'))
        self.item2id = cPickle.load(open(item2id_path,'w'))

        # TODO: initialize vectorizer
        self.vectorizer = Vectorizer()

    def load_lightfm_matrix(self, lightfm_path):
        assert os.path.exists(lightfm_path)
        lightfm = cPickle.load(open(lightfm_path, 'w'))

        u_pref_matrix = lightfm.user_embeddings.astype('float32')
        self.u_pref_zero_matrix = np.vstack([u_pref_matrix, np.zeros_like(u_pref_matrix[0, :])])

        v_pref_matrix = lightfm.item_embeddings.astype('float32')
        self.v_pref_zero_matrix = np.vstack([v_pref_matrix, np.zeros_like(v_pref_matrix[0, :])])

        u_bias_matrix = lightfm.user_biases.reshape((-1,1)).astype('float32')
        v_bias_matrix = lightfm.item_biases.reshape((-1,1)).astype('float32')

        self.u_bias_zero_matrix = np.vstack([u_bias_matrix, np.zeros_like(u_bias_matrix[0,:])])
        self.v_bias_zero_matrix = np.vstack([v_bias_matrix, np.zeros_like(v_bias_matrix[0,:])])

    def predict(self, user_profiles ):
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
            self.model.u_cont: u_cont,
            self.model.v_cont: v_cont,
            self.model.phase: 0
        }

        topk_col_ids = self.model.sess.run(tf.nn.top_k(self.model.predicted_R, k=n_items, sorted=True, name='predicted_topk'),
                            ip_feed_dict)
        return topk_col_ids


if __name__ == '__main__':

    pass