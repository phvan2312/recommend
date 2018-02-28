from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
import numpy as np
import pandas as pd
import cPickle

class ProfileVectorizer:
    def __init__(self, max_feature):
        self.max_feature = max_feature
        self.tfidfs = {}

    def train(self, datas):
        # for training, datas variable must be 4 (basics,education,skills,work) list of string
        for k, v in datas.items():
            self.tfidfs[k] = TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english', max_features=self.max_feature,
                                     ngram_range=(1, 1))
            self.tfidfs[k].fit(v)

    def predict(self, datas):
        # for inference, datas variable must be 4 (basics,education,skills,work) list of string
        results = OrderedDict((k,self.tfidfs[k].transform(v).toarray()) for k,v in datas.items())
        return np.hstack([v for _,v in results.items()]).astype('float32')

if __name__ == '__main__':
    # load training dataset
    profile_detail_path = './../metadata/profile.csv'
    profile_detail = pd.read_csv(profile_detail_path, sep=',')
    profile_detail.fillna('',inplace=True)
    training_data = {
        'education': profile_detail['education'].tolist(),
        'skills': profile_detail['skills'].tolist(),
        'work': profile_detail['work'].tolist(),
    }

    # vectorizer
    profile_vect = ProfileVectorizer(max_feature=100)
    profile_vect.train(training_data)
    vectorizered_profile_detail = profile_vect.predict(training_data)

    # save
    cPickle.dump(profile_vect, open('./profile_vectorizer.pkl','w'))
    vectorizered_profile_detail.tofile(open('./profile.csv.bin', 'w'))