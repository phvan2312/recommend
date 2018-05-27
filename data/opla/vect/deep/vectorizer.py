"""
Usage: assume that we're in deep folder
python vectorizer.py --item_detail_path=./../../metadata/out/item.csv --batch_size=32 --n_epochs=10 --class_path=./saved_model/model.pkl --saved_path=./saved_model/deepvec.ckpt
--out_item_vect_path=./out/item.csv.bin --profile_detail_path=./../../metadata/out/profile.csv --out_profile_vect_path=./out/profile.csv.bin
"""
"""
--out_item_vect_path',help='path storing item vectors, empty if do not want',dest='out_item_vect_path',type=str)
parser.add_argument('--profile_detail_path',help='path containing profile information',dest='profile_detail_path',type=str)
parser.add_argument('--out_profile_vect_path'
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import cPickle
import json
import argparse

def extract_work_from_item(work_desc):
    works = [[w['work'] for w in work] for work in json.loads(work_desc)]
    unique_works = set()
    for work in works:
        for _w in work: unique_works.add(_w)

    return list(unique_works)

def extract_skill_from_item(skill_desc):
    skills = [[s['skill'] for s in work] for work in json.loads(skill_desc)]
    unique_skills = set()
    for skill in skills:
        for _s in skill: unique_skills.add(_s)

    return list(unique_skills)

def extract_work_from_user(work_desc):
    return [w['position'] for w in json.loads(work_desc)]

def extract_skill_from_user(skill_desc):
    return [s['skill'] for s in json.loads(skill_desc)]

def build_frequency(texts):
    assert type(texts) is list
    dct = {}

    for _texts in texts:
        for text in _texts:
            keys = text.strip().lower()

            for key in keys.split(' '):
                if key in dct: dct[key] += 1
                else: dct[key] = 1

    return dct

def build_vocab(texts):
    freq_dict = build_frequency(texts)

    id2x = {k:v for k,v in enumerate(['<unk>','<pad>'] + freq_dict.keys())}
    x2id = {v:k for k,v in id2x.items()}

    return id2x, x2id

def mapping(word2id,text):
    return [word2id[token if token in word2id else '<unk>'] for token in text.lower().split(' ')]

def random_mapping(word2id, text_len):
    vocab_len = len(word2id)
    return [np.random.randint(0,vocab_len) for _ in xrange(text_len)]

class VectorizerModel:
    def __init__(self,item_dim,profile_dim,n_item,n_profile,lr,max_sen_len,dropout_prob,word2id):
        self.item_dim = item_dim
        self.profile_dim = profile_dim

        self.n_item = n_item
        self.n_profile = n_profile

        self.lr = lr
        self.max_sen_len = max_sen_len
        self.dropout_prob = dropout_prob

        self.word2id = word2id

    def __build_embedding(self, n_item, n_profile, item_dim, profile_dim):
        item_emb = tf.get_variable(name='item_emb', shape=(n_item, item_dim), initializer=tf.contrib.layers.xavier_initializer())
        profile_emb = tf.get_variable(name='profile_emb', shape=(n_profile, profile_dim), initializer=tf.contrib.layers.xavier_initializer())

        return item_emb, profile_emb

    def __build_placeholder(self):
        # for item
        item_ph  = tf.placeholder(dtype=tf.int32,shape=(None,), name='item_placeholder')

        # for profile
        skill_ph = tf.placeholder(dtype=tf.int32,shape=(None,None), name='skill_placeholder')
        work_ph  = tf.placeholder(dtype=tf.int32,shape=(None,None), name='work_placeholder')

        return item_ph, skill_ph, work_ph

    def __build_one_layer(self,input,indim,outdim,active_func = lambda x:x):
        W = tf.get_variable(name='W',shape=(indim,outdim),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b',shape=(outdim),dtype=tf.float32,initializer=tf.zeros_initializer())

        z = tf.nn.xw_plus_b(input,W,b)
        a = active_func(z)

        return a

    def get_profile_vector(self,works,skills):
        assert type(works) is list
        assert type(skills) is list

        work_ids  = [[mapping(self.word2id,work) for work in _works] if len(_works) > 0 else [[1]] for _works in works]
        skill_ids = [[mapping(self.word2id,skill) for skill in _skills] if len(_skills) > 0 else [[1]] for _skills in skills]

        work_phs = [pad_common(sequences=work_id, pad_tok=1, max_length=self.max_sen_len)[0] for work_id in work_ids]
        skill_phs = [pad_common(sequences=skill_id, pad_tok=1, max_length=self.max_sen_len)[0] for skill_id in skill_ids]

        padding_work_phs  = pad_common(sequences=work_phs, pad_tok=[len(self.word2id)] * self.max_sen_len,max_length=10)[0]
        padding_skill_phs = pad_common(sequences=skill_phs,pad_tok=[len(self.word2id)] * self.max_sen_len,max_length=10)[0]

        work_vect = self.sess.run(self.work_vects,
                                     feed_dict={
                                         self.work_phs: padding_work_phs,
                                     })

        skill_vect = self.sess.run(self.skill_vects,
                                 feed_dict={
                                     self.skill_phs: padding_skill_phs,
                                 })

        return np.concatenate((work_vect,skill_vect),axis=1)

    def get_item_vector(self,item_id):
        return self.sess.run(tf.gather(self.item_emb,indices=item_id),feed_dict={})

    def build(self):
        # build placeholder
        with tf.variable_scope('build_placeholder'):
            self.item_ph, self.skill_ph, self.work_ph = self.__build_placeholder()
            self.skill_target_ph =  tf.placeholder(dtype=tf.float32,shape=(None,), name='skill_target_placeholder')
            self.work_target_ph = tf.placeholder(dtype=tf.float32,shape=(None,), name='work_target_placeholder')
            self.dropout_prob_ph = tf.placeholder_with_default(self.dropout_prob,shape=(),name='dropout_prob_placeholder')

        # build matrix embedding
        with tf.variable_scope('build_embedding_matrix'):
            self.item_emb, self.profile_emb = self.__build_embedding(n_item=self.n_item,n_profile=self.n_profile,
                                                                     item_dim=self.item_dim,profile_dim=self.profile_dim)

        # get embedded vector
        with tf.variable_scope('embedding_matrix'):
            self.item_vect = tf.nn.embedding_lookup(self.item_emb, self.item_ph) #(item_dim)
            self.skill_vect = tf.nn.embedding_lookup(self.profile_emb, self.skill_ph) #(n_words,word_dim)
            self.work_vect = tf.nn.embedding_lookup(self.profile_emb, self.work_ph) #(n_words,word_dim)

        # get {skill,work} vector by sum up its word vector.
        self.skill_vect = tf.reduce_mean(self.skill_vect,axis=1,name='mean_skill_vect') #(word_dim)
        self.work_vect  = tf.reduce_mean(self.work_vect,axis=1,name='mean_work_vect') #(word_dim)

        item_skill_vect = tf.concat([self.item_vect,self.skill_vect],axis=1, name='concat_item_skill')
        work_skill_vect = tf.concat([self.item_vect,self.work_vect], axis=1, name='concat_work_skill')

        item_skill_vect = tf.nn.dropout(item_skill_vect, keep_prob=self.dropout_prob_ph)
        work_skill_vect = tf.nn.dropout(work_skill_vect, keep_prob=self.dropout_prob_ph)

        # build one layer neural netwok
        with tf.variable_scope("build_layer"):
            with tf.variable_scope("layer_skill"):
                self.predicted_skill = self.__build_one_layer(input=item_skill_vect,indim=self.item_dim+self.profile_dim,
                                                              outdim=1)

            with tf.variable_scope("layer_work"):
                self.predicted_work  = self.__build_one_layer(input=work_skill_vect,indim=self.item_dim+self.profile_dim,
                                                              outdim=1)

        # build loss
        with tf.variable_scope('loss'):
            skill_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.skill_target_ph,logits=tf.reshape(self.predicted_skill,shape=(-1,))
                                                                 ,name='skill_loss')
            work_loss  = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.work_target_ph,logits=tf.reshape(self.predicted_work,shape=(-1,))
                                                                 ,name='work_loss')
            self.fn_loss = tf.reduce_mean(skill_loss + work_loss)

        # build optimizer
        with tf.variable_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.fn_loss)

        # buil some materials for faster vectorizer
        self.skill_phs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='skill_placeholders') #(n_user,n_skills,n_tokens)
        self.work_phs  = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='work_placeholders')

        self.profile_emb_with_zero_last = tf.concat([self.profile_emb,tf.zeros((1,self.profile_dim),dtype=tf.float32)],axis=0)

        skill_vects = tf.nn.embedding_lookup(self.profile_emb_with_zero_last,self.skill_phs) #(n_user,n_skills,n_tokens,n_dim)
        work_vects  = tf.nn.embedding_lookup(self.profile_emb_with_zero_last,self.work_phs)

        self.skill_vects = tf.reduce_mean(tf.reduce_mean(skill_vects,axis=2), axis=1)
        self.work_vects  = tf.reduce_mean(tf.reduce_mean(work_vects, axis=2), axis=1)

        # build session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)

    def save(self,save_path):
        self.saver.save(self.sess,save_path)

    def load(self,save_path):
        self.saver.restore(sess=self.sess,save_path=save_path)

    def run(self, batch, mode='train'):

        ip_feed_dct = {
            self.item_ph: [e['item_id'] for e in batch],
            self.skill_ph: pad_common(sequences=[e['skill'] for e in batch], pad_tok=1, max_length=self.max_sen_len)[0],
            self.work_ph: pad_common(sequences=[e['work_pos'] for e in batch], pad_tok=1, max_length=self.max_sen_len)[0],
            self.skill_target_ph: [e['skill_point'] for e in batch],
            self.work_target_ph: [e['work_pos_point'] for e in batch],
        }

        if mode == 'train':
            _, loss = self.sess.run([self.train_op,self.fn_loss],ip_feed_dct)

            return loss

        if mode == 'inference':
            ip_feed_dct[self.dropout_prob_ph] = 1.0

            predicted_skill_proba = tf.nn.sigmoid(self.predicted_skill)
            predicted_work_proba = tf.nn.sigmoid(self.predicted_work)

            s,w = self.sess.run([predicted_skill_proba,predicted_work_proba],ip_feed_dct)

            return (s,w)

def pad_common(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

# saved_path = './saved_model/deepvec.ckpt'
# class_path = './saved_model/model.pkl'
# item_vect_path = './item.csv.bin'
# profile_vect_path = './profile.csv.bin'

parser  = argparse.ArgumentParser()

parser.add_argument('--item_detail_path',help='path containing item information', dest='item_detail_path',type=str)
parser.add_argument('--batch_size',help='number of samples per batch',dest='batch_size',type=int)
parser.add_argument('--n_epochs',help='number of epochs',dest='n_epochs',type=int)
parser.add_argument('--class_path',help='path for storing class',dest='class_path',type=str)
parser.add_argument('--saved_path',help='path for storing weighted matrix',dest='saved_path',type=str)
parser.add_argument('--out_item_vect_path',help='path storing item vectors, empty if do not want',dest='out_item_vect_path',type=str)
parser.add_argument('--profile_detail_path',help='path containing profile information',dest='profile_detail_path',type=str)
parser.add_argument('--out_profile_vect_path',help='path storing profile vectors, empty if do not want',dest='out_profile_vect_path',type=str)

args    = vars(parser.parse_known_args()[0])

def train(item_detail_path,batch_size,n_epochs,class_path,saved_path,**kargs):
    # load data from file
    item_detail = pd.read_csv(item_detail_path, sep=',')
    item_detail.fillna('', inplace=True)
    print ('loaded %d items from %s ...' % (item_detail.shape[0], item_detail_path))


    # extract work and skills
    item_extract_works = [extract_work_from_item(work_desc) for work_desc in item_detail['work'].tolist()]
    item_extract_skills = [extract_skill_from_item(skill_desc) for skill_desc in item_detail['skill'].tolist()]

    # building vocabulary
    id2word, word2id = build_vocab(item_extract_skills + item_extract_works)

    # create dataset
    assert len(item_extract_works) == len(item_extract_skills)

    datasets = []
    for item_id, (works, skills) in enumerate(zip(item_extract_works, item_extract_skills)):
        dataset = {}

        dataset['item_id'] = item_id
        for work in works:
            sample = dataset.copy()
            sample['work_pos'] = mapping(word2id,work)
            sample['skill'] = random_mapping(word2id,np.random.randint(3,6))
            sample['work_pos_point'] = 1.0
            sample['skill_point'] = 0.0

            datasets.append(sample)

        for skill in skills:
            sample = dataset.copy()
            sample['skill'] = mapping(word2id,skill)
            sample['work_pos'] = random_mapping(word2id,np.random.randint(3,6))
            sample['work_pos_point'] = 0.0
            sample['skill_point'] = 1.0

            datasets.append(sample)

    random.shuffle(datasets)
    print ('created dataset, with %d samples ...' % len(datasets))

    train_datasets, test_datasets = datasets, []

    # create batchs
    batch_ids = [(batch_start, min(batch_start + 32, len(datasets))) for batch_start in
                 range(0, len(train_datasets), batch_size)]

    batchs = []
    for start, end in batch_ids: batchs.append(train_datasets[start:end])
    print ('created batches, with batch_size: %d, have %d batchs ...' % (batch_size,len(batchs)))

    # build model
    model = VectorizerModel(item_dim=200, profile_dim=100, n_item=item_detail.shape[0], n_profile=len(word2id),
                            lr=0.001, max_sen_len=5, dropout_prob=0.6, word2id=word2id)
    cPickle.dump(model, open(class_path, 'w'))
    print ('saved class model to %s ...' % class_path)

    model.build()

    # training
    print ('start training ...')
    for epoch_id in range(n_epochs):
        all_loss = []

        for batch_id in np.random.permutation(range(len(batchs))):
            cur_batch = batchs[batch_id]
            loss = model.run(cur_batch)

            all_loss += [loss]

        print 'e%i_loss: %.4f' % (epoch_id + 1, np.mean(all_loss))

    # saving

    model.save(save_path=saved_path)
    print ('saved weight model to %s ...' % saved_path)

    return model

def extract_item_vector(model, item_detail_path, out_item_vect_path,**kargs):
    item_detail = pd.read_csv(item_detail_path, sep=',')
    item_detail.fillna('', inplace=True)

    n_items = item_detail.shape[0]

    item_vects = model.get_item_vector(item_id=range(n_items))
    print ('item_vects shape: ', item_vects.shape)
    item_vects.tofile(open(out_item_vect_path, 'w'))

    print ('saved item_vectors to file %s ...' % out_item_vect_path)

def extract_profile_vector(model, profile_detail_path, out_profile_vect_path,**kargs):
    profile_detail = pd.read_csv(profile_detail_path, sep=',')
    profile_detail.fillna('', inplace=True)

    profile_works = [extract_work_from_user(work_desc) for work_desc in profile_detail['work'].tolist()]
    profile_skills = [extract_skill_from_user(skill_desc) for skill_desc in profile_detail['skills'].tolist()]

    assert len(profile_works) == len(profile_skills)
    profile_vects = model.get_profile_vector(works=profile_works, skills=profile_skills)

    print ('profile_vects shape: ', profile_vects.shape)

    profile_vects.tofile(open(out_profile_vect_path, 'w'))
    print ('saved profile_vectors to file %s ...' % out_profile_vect_path)

if __name__ == '__main__':
    model = train(**args)

    if args['out_item_vect_path'] != '':
        extract_item_vector(model=model,item_detail_path=args['item_detail_path'],out_item_vect_path=args['out_item_vect_path'])

    if args['out_profile_vect_path'] != '':
        extract_profile_vector(model=model,profile_detail_path=args['profile_detail_path'],out_profile_vect_path=args['out_profile_vect_path'])
