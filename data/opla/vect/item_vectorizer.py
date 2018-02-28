from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import cPickle

class ItemVectorizer:
    def __init__(self,max_feature):
        self.max_feature = max_feature
        self.tfidf = TfidfVectorizer(min_df=2,max_df=0.9,stop_words='english',max_features=max_feature,ngram_range=(1,1))

    def train(self,datas):
        # for training, datas variable must be a list of string
        self.tfidf.fit(datas)

    def predict(self,datas):
        # for inference, datas variable must be a list of string
        return self.tfidf.transform(datas).toarray().astype('float32')

if __name__ == '__main__':
    # load training dataset
    item_detail_path = './../metadata/item.csv'
    item_detail = pd.read_csv(item_detail_path,sep=',')
    item_detail.fillna('',inplace=True)
    training_data = item_detail['desc'].tolist()

    # vectorizer
    item_vect = ItemVectorizer(max_feature=200)
    item_vect.train(training_data)
    vectorizered_item_detail = item_vect.predict(training_data)

    # save
    cPickle.dump(item_vect, open('./item_vectorizer.pkl','w'))
    vectorizered_item_detail.tofile(open('./item.csv.bin','w'))