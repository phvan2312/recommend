import tensorflow as tf
import pandas as pd
import re
import numpy as np
import random

def extract_work(texts, regex = ur"work:\s((?!(,\splace:))\w(\s)?)+"):
    all = []

    for text in texts:
        matches = re.finditer(regex, text)
        all.append([match.group().replace('work: ','') for match in matches])

    return all

def extract_skill(texts, regex = ur"skill:\s((?!(,\s(place|work):))\w(\s)?)+"):
    all = []

    for text in texts:
        matches = re.finditer(regex, text)
        all.append([match.group().replace('skill: ', '') for match in matches])

    return all

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

    id2x = {k:v for k,v in enumerate(['<unk>'] + freq_dict.keys())}
    x2id = {v:k for k,v in id2x.items()}

    return id2x, x2id

def mapping(word2id,text):
    return [word2id[token if token in word2id else '<unk>'] for token in text.lower().split(' ')]

def random_mapping(word2id, text_len):
    vocab_len = len(word2id)
    return [np.random.randint(0,vocab_len) for _ in xrange(text_len)]

class ItemVectorizerModel:
    def __init__(self,item_dim,profile_dim,n_item,n_profile,lr,max_sen_len):
        self.item_dim = item_dim
        self.profile_dim = profile_dim

        self.n_item = n_item
        self.n_profile = n_profile

        self.lr = lr
        self.max_sen_len = max_sen_len

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

    def build(self):

        # build placeholder
        with tf.variable_scope('build_placeholder'):
            self.item_ph, self.skill_ph, self.work_ph = self.__build_placeholder()
            self.skill_target_ph =  tf.placeholder(dtype=tf.float32,shape=(None,), name='skill_target_placeholder')
            self.work_target_ph = tf.placeholder(dtype=tf.float32,shape=(None,), name='work_target_placeholder')

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
        self.skill_vect = tf.reduce_sum(self.skill_vect,axis=1,name='sum_skill_vect') #(word_dim)
        self.work_vect  = tf.reduce_sum(self.work_vect,axis=1,name='sum_work_vect') #(word_dim)

        item_skill_vect = tf.concat([self.item_vect,self.skill_vect],axis=1, name='concat_item_skill')
        work_skill_vect = tf.concat([self.item_vect,self.work_vect], axis=1, name='concat_work_skill')

        # build one layer neural netwok
        with tf.variable_scope("build_layer"):
            predicted_skill = self.__build_one_layer(input=item_skill_vect,indim=self.item_dim+self.profile_dim,outdim=1)
            predicted_work  = self.__build_one_layer(input=work_skill_vect,indim=self.item_dim+self.profile_dim,outdim=1)

        # build loss
        with tf.variable_scope('loss'):
            skill_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.skill_target_ph,logits=predicted_skill
                                                                 ,name='skill_loss')
            work_loss  = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.work_target_ph,logits=predicted_work
                                                                 ,name='skill_loss')
            self.fn_loss = skill_loss + work_loss

        # build optimizer
        with tf.variable_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.fn_loss)

        # build session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)

    def run(self, batch):

        ip_feed_dct = {
            self.item_ph: [e['item_ph'] for e in batch],
            self.skill_ph: pad_common(sequences=[e['skill'] for e in batch], pad_tok=1, max_length=self.max_sen_len)[0],
            self.work_ph: pad_common(sequences=[e['work_pos'] for e in batch], pad_tok=1, max_length=self.max_sen_len)[0],
            self.skill_target_ph: [e['skill_pos_point'] for e in batch],
            self.work_target_ph: [e['work_pos_point'] for e in batch],
        }

        _, loss = self.sess.run([self.train_op,self.fn_loss],ip_feed_dct)

        return loss

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

if __name__ == '__main__':
    item_detail_path = './../../metadata/item.csv'
    item_detail = pd.read_csv(item_detail_path, sep=',')
    item_detail.fillna('', inplace=True)

    item_descs = item_detail['desc'].tolist()
    item_extract_works  = extract_work(texts=item_descs)
    item_extract_skills = extract_skill(texts=item_descs)

    id2word, word2id = build_vocab(item_extract_skills + item_extract_works)

    # create dataset
    assert len(item_extract_works) == len(item_extract_skills)
    datasets = []

    for item_id, (works, skills) in enumerate(zip(item_extract_works, item_extract_skills)):
        dataset = {}

        dataset['item_id'] = item_id
        for work in works:
            tmp_dataset = dataset.copy()
            tmp_dataset['work_pos'] = mapping(word2id,work)
            tmp_dataset['skill'] = random_mapping(word2id,np.random.randint(3,6))
            tmp_dataset['work_pos_point'] = 1.0
            tmp_dataset['skill_point'] = 0.0

            datasets.append(tmp_dataset)

        for skill in skills:
            tmp_dataset = dataset.copy()
            tmp_dataset['skill'] = mapping(word2id,skill)
            tmp_dataset['work_pos'] = random_mapping(word2id,np.random.randint(3,6))
            tmp_dataset['work_pos_point'] = 0.0
            tmp_dataset['skill_point'] = 1.0

            datasets.append(tmp_dataset)
        pass

    random.shuffle(datasets)

    # create batchs
    batch_size = 32
    batch_ids = [(batch_start, min(batch_start + 32,len(datasets))) for batch_start in range(0,len(datasets),batch_size)]

    batchs = []
    for start, end in batch_ids: batchs.append(datasets[start:end])

    # build model
    model = ItemVectorizerModel(item_dim=200,profile_dim=100,n_item=len(item_descs),n_profile=len(word2id),lr=0.001,
                                max_sen_len=5)
    model.build()

    # training
    n_epochs = 10

    for epoch_id in enumerate(range(n_epochs)):
        for batch_id in np.random.permutation(range(len(batchs))):
            cur_batch = batchs[batch_id]
            loss = model.run(cur_batch)



            pass


    pass