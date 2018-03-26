from collections import OrderedDict
from scipy.sparse import coo_matrix
import numpy as np

class EvalBatchSample:
    def __init__(self, user_ids, item_ids, ratings, batch_size, u_pref, v_pref, u_cont, v_cont,u_bias,v_bias):
        self.user_ids = user_ids.astype(np.int32)
        self.item_ids = item_ids.astype(np.int32)
        self.ratings  = ratings.astype(np.float32)

        vocab_user_ids = list(set(self.user_ids))
        vocab_item_ids = list(set(self.item_ids))

        self.mapping_user_ids = OrderedDict({e: i for i, e in enumerate(vocab_user_ids)})
        self.mapping_item_ids = OrderedDict({e: i for i, e in enumerate(vocab_item_ids)})

        self.rows = [self.mapping_user_ids[user_id] for user_id in self.user_ids]
        self.cols = [self.mapping_item_ids[item_id] for item_id in self.item_ids]

        self.target = coo_matrix(
            (
                np.ones(len(self.user_ids)),
                (self.rows, self.cols)
            ),
            shape=(len(self.mapping_user_ids), len(self.mapping_item_ids))
        ).toarray()

        self.u_pref = u_pref[vocab_user_ids, :]
        self.v_pref = v_pref[vocab_item_ids, :]
        self.u_cont = u_cont[vocab_user_ids, :]
        self.v_cont = v_cont[vocab_item_ids, :]
        self.u_bias = u_bias[vocab_user_ids, :]
        self.v_bias = v_bias[vocab_item_ids, :]

        self.__create_batchs(batch_size=batch_size)

    def __create_batchs(self, batch_size):
        max_user = self.target.shape[0]
        self.eval_batchs = [(x, min(x + batch_size, max_user)) for x
                           in xrange(0, max_user, batch_size)]

    def get_batch(self, batch_id):
        batch_start, batch_end = self.eval_batchs[batch_id]
        feed_dict = {
            'u_pref': self.u_pref[batch_start:batch_end],
            'v_pref': self.v_pref,
            'u_cont': self.u_cont[batch_start:batch_end],
            'v_cont': self.v_cont,
            'u_bias': self.u_bias[batch_start:batch_end],
            'v_bias': self.v_bias
        }

        return feed_dict

class TrainBatchSample:
    dropped_user_type = 'dropped_user_type'
    dropped_item_type = 'dropped_item_type'

    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids.astype(np.int32)
        self.item_ids = item_ids.astype(np.int32)
        self.ratings  = ratings.astype(np.float32)

        vocab_user_ids = list(set(self.user_ids))
        vocab_item_ids = list(set(self.item_ids))

        self.mapping_user_ids = OrderedDict([(e,i) for i,e in enumerate(vocab_user_ids)])
        self.mapping_item_ids = OrderedDict([(e,i) for i,e in enumerate(vocab_item_ids)])

        #
        # need for placeholder
        #
        self.u_pref_row_ids = vocab_user_ids
        self.u_content_row_ids = vocab_user_ids
        self.v_pref_row_ids = vocab_item_ids
        self.v_content_row_ids = vocab_item_ids
        self.u_bias_row_ids = vocab_user_ids
        self.v_bias_row_ids = vocab_item_ids

        self.rows = [self.mapping_user_ids[user_id] for user_id in self.user_ids]
        self.cols = [self.mapping_item_ids[item_id] for item_id in self.item_ids]

        self.row_col_ids = np.array([(row,col) for (row,col) in zip(self.rows,self.cols)])

        self.target = coo_matrix(
            (
                self.ratings,
                (self.rows,self.cols)
            ),
            shape=(len(self.mapping_user_ids), len(self.mapping_item_ids))
        ).toarray()

    def __dropout_X(self, dropout_prob, org_ids, zero_id, dropped_type):
        assert dropped_type in [self.dropped_user_type, self.dropped_item_type]

        n_X = len(org_ids)

        drop_len = int(n_X * dropout_prob)
        drop_ids = np.random.permutation(n_X)[:drop_len]

        for id in drop_ids: org_ids[id] = zero_id

    def dropout_user(self, dropout_prob, zero_id):
        self.u_pref_row_ids = self.mapping_user_ids.keys()
        self.__dropout_X(dropout_prob, self.u_pref_row_ids, zero_id, self.dropped_user_type)

    def dropout_item(self, dropout_prob, zero_id):
        self.v_pref_row_ids = self.mapping_item_ids.keys()
        self.__dropout_X(dropout_prob, self.v_pref_row_ids, zero_id, self.dropped_item_type)
