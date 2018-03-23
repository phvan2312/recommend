import sys
sys.path.append('..')

import tensorflow as tf
from batch import TrainBatchSample, EvalBatchSample

class RecommendNet:
    def __init__(self, latent_dim, user_feature_dim, item_feature_dim, out_dim, multilayer_dims = [400], do_batch_norm = True,
                 default_lr = 0.005, k = 50,
                 np_u_pref_scaled = None, np_v_pref_scaled = None, np_u_cont = None, np_v_cont = None, **kargs):

        self.latent_dim = latent_dim
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.out_dim = out_dim

        self.multilayer_dims = multilayer_dims
        self.do_batch_norm = do_batch_norm

        self.np_u_pref_scaled = np_u_pref_scaled
        self.np_v_pref_scaled = np_v_pref_scaled
        self.np_u_cont = np_u_cont
        self.np_v_cont = np_v_cont

        self.k = k
        self.default_lr = default_lr
        self.train_signal, self.inf_signal = 'training', 'inference'



    def get_params(self):
        params = {
            'latent_dim' : self.latent_dim,
            'user_feature_dim' : self.user_feature_dim,
            'item_feature_dim' : self.item_feature_dim,
            'out_dim' : self.out_dim,
            'multilayer_dims' : self.multilayer_dims,
            'do_batch_norm' : self.do_batch_norm,
            'default_lr' : self.default_lr,
            'k' : self.k,
            'np_u_pref_scaled' : None,
            'np_v_pref_scaled' : None,
            'np_u_cont' : None,
            'np_v_cont' : None
        }

        return params

    def batch_normalize(self, data, scope, phase):
        assert hasattr(self,'phase')

        norm_data = tf.contrib.layers.batch_norm(
            data,
            decay=0.9,
            center=True,
            scale=True,
            is_training=phase,
            scope=scope + '_bn')

        return norm_data

    def __build_multi_layer(self, data, init_dim, multi_layer_dims, do_batch_norm):
        last_data = data
        all_dims = [init_dim]

        for i, multilayer_dim in enumerate(multi_layer_dims):
            with tf.variable_scope("multilayer_dim%s_%s" % (multilayer_dim, i)):
                last_dim = all_dims[-1]

                w = tf.get_variable(name='w', shape=[last_dim, multilayer_dim], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name='b', shape=[multilayer_dim], dtype=tf.float32,
                                      initializer=tf.zeros_initializer())

                last_data = tf.nn.xw_plus_b(last_data, w, b)
                if do_batch_norm: last_data = self.batch_normalize(last_data,scope='_',phase=self.phase)

                last_data = tf.nn.tanh(last_data)

                # keep track last dim
                all_dims.append(multilayer_dim)

        return all_dims[-1], last_data

    def __build_fully_connected(self,data,dim_in,dim_out):
        w = tf.get_variable(name='w', shape=[dim_in,dim_out],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', shape=[dim_out], dtype=tf.float32, initializer=tf.zeros_initializer())

        return tf.nn.xw_plus_b(data,w,b)

    def __build_placeholder(self):
        self.u_pref = tf.placeholder(dtype=tf.float32, shape=[None,self.latent_dim], name='u_pref') # (n_users, laten_dim)
        self.v_pref = tf.placeholder(dtype=tf.float32, shape=[None,self.latent_dim], name='v_pref') # (n_items, latent_dim)
        self.u_content = tf.placeholder(dtype=tf.float32, shape=[None,self.user_feature_dim], name='u_content') # (n_users, user_feature_dim)
        self.v_content = tf.placeholder(dtype=tf.float32, shape=[None,self.item_feature_dim],name='v_content') # (n_items, item_feature_dim)

        if self.do_batch_norm: self.phase = tf.placeholder(tf.bool, name='phase')

        self.topk = tf.placeholder_with_default(input=self.k,shape=[],name='topk')
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, None], name='target')
        self.row_col_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='row_col_ids')
        self.lr = tf.placeholder(dtype=tf.float32,shape=[],name='lr')

    def __build_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        return sess, saver

    def build(self, build_session = True):

        with tf.variable_scope('build_placeholder'):
            self.__build_placeholder()

        with tf.variable_scope('build_multilayer'):
            init_dims = (self.latent_dim + self.user_feature_dim, self.latent_dim + self.item_feature_dim) #(u_last_dim,v_last_dim)

            self.last_u = tf.concat([self.u_pref,self.u_content], axis = 1, name='concat_u')
            self.last_v = tf.concat([self.v_pref,self.v_content], axis = 1, name='concat_v')

            with tf.variable_scope('user'):
                last_u_dim, self.last_u = self.__build_multi_layer(data=self.last_u, init_dim=init_dims[0],
                                                                   multi_layer_dims=self.multilayer_dims,
                                                                   do_batch_norm=self.do_batch_norm)

            with tf.variable_scope('item'):
                last_v_dim, self.last_v = self.__build_multi_layer(data=self.last_v, init_dim=init_dims[1],
                                                                   multi_layer_dims=self.multilayer_dims,
                                                                   do_batch_norm=self.do_batch_norm)


        with tf.variable_scope('build_fully_connected'):
            with tf.variable_scope('user'):
                self.u_hat = self.__build_fully_connected(data=self.last_u,dim_in=last_u_dim,dim_out=self.out_dim)

            with tf.variable_scope('item'):
                self.v_hat = self.__build_fully_connected(data=self.last_v,dim_in=last_v_dim,dim_out=self.out_dim)


        with tf.variable_scope('build_loss'):
            self.predicted_R = tf.matmul(self.u_hat,self.v_hat,transpose_b=True,name='predicted_R')

            nonzero_predicted_R = tf.gather_nd(self.predicted_R, self.row_col_ids)
            nonzero_target = tf.gather_nd(self.target, self.row_col_ids)

            self.loss = tf.reduce_mean(tf.squared_difference(nonzero_predicted_R, nonzero_target))

        with tf.variable_scope('build_eval'):
            cond_k = tf.cond(self.topk <= tf.shape(self.predicted_R)[-1],lambda: self.topk, lambda: tf.shape(self.predicted_R)[-1])

            _,self.predicted_topk = tf.nn.top_k(self.predicted_R,k=cond_k,sorted=True,name='predicted_topk')
            _,self.real_topk = tf.nn.top_k(self.target,k=cond_k,sorted=True,name='real_topk')

            predicted_topk = tf.reshape(self.predicted_topk, shape=[-1])
            real_topk = tf.reshape(self.real_topk, shape=[-1])

            correct_prediction = tf.equal(predicted_topk, real_topk)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32),name='accuracy')

        with tf.variable_scope('build_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        if build_session:
            self.sess, self.saver = self.__build_session()

    def __check_correct_data(self, datas, mode):
        assert isinstance(datas, TrainBatchSample) or isinstance(datas,EvalBatchSample)
        assert mode in [self.train_signal, self.inf_signal]



    def create_input(self, datas, mode='training',**kargs):
        self.__check_correct_data(datas,mode)

        def create_train_input(datas,**kargs):
            assert isinstance(datas, TrainBatchSample)

            feed_dict = {
                self.u_pref: self.np_u_pref_scaled[datas.u_pref_row_ids, :],
                self.v_pref: self.np_v_pref_scaled[datas.v_pref_row_ids, :],
                self.u_content: self.np_u_cont[datas.u_content_row_ids],
                self.v_content: self.np_v_cont[datas.v_content_row_ids],
                self.target: datas.target,
                self.lr: kargs['lr'] if 'lr' in kargs else self.default_lr,
                self.row_col_ids: datas.row_col_ids,
                self.phase: 1 if self.do_batch_norm else 0
            }

            return feed_dict

        def create_eval_input(datas,**kargs):
            assert isinstance(datas, EvalBatchSample)
            assert 'batch_id' in kargs

            ips = datas.get_batch(kargs['batch_id'])

            feed_dict = {
                self.u_pref: ips['u_pref'],
                self.v_pref: ips['v_pref'],
                self.u_content: ips['u_cont'],
                self.v_content: ips['v_cont'],
                self.phase: 0
            }

            return feed_dict

        if mode == self.train_signal: return create_train_input(datas, **kargs)
        if mode == self.inf_signal: return create_eval_input(datas, **kargs)

    def save(self,path):
        saved_path = self.saver.save(self.sess,path)
        return saved_path

    def restore(self,path):
        self.saver.restore(self.sess,path)

    def run(self, datas, mode='training',**kargs):
        self.__check_correct_data(datas, mode)

        ip_feed_dict = self.create_input(datas,mode, **kargs)

        if mode == self.train_signal:
            try:
                arg1, arg2, _ = self.sess.run([self.loss, self.predicted_R, self.train_op], feed_dict=ip_feed_dict)
            except:
                print 'ahihi'
        elif mode == self.inf_signal:
            arg2 = None
            predicted_topk = self.sess.run(self.predicted_topk, feed_dict=ip_feed_dict)
            arg1 = predicted_topk

        return arg1, arg2
