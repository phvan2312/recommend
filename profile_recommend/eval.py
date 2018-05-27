import cPickle
import argparse

from nn.isgd import ISGD

parser = argparse.ArgumentParser(description='... PROFILE RECOMMENDATION ...')

parser.add_argument('--save_path' ,help='materials path for loadding old model' ,dest='save_path',type=str,default='')
parser.add_argument('--users'  ,help='list of user, seperated by a comma', dest='users',type=str,default='')
parser.add_argument('--topk',help='get only k items for evaluating',dest='topk',type=int)

args   = parser.parse_args()

def eval(save_path, users, topk):
    # load model from save_path
    model, user2id = cPickle.load(open(save_path, 'r'))
    print ('loaded model from %s ...' % save_path)

    # building id2user conversely
    id2user = {i:u for u,i in user2id.items()}

    # convert original name to their mapping ids
    choosen_user = filter(lambda u: u in model.known_users, users.split(','))
    print ("preprocessed, some user may be ignored if they're in cold_start ...")
    choosen_user = map(lambda u: user2id[u], choosen_user)

    # predict
    predict_ids  = model.recommends(choosen_user, topk) # type of [[],[],...]
    result = {}

    for user,user_predict_ids in zip(choosen_user,predict_ids):
        result[user] = [id2user[i] for i in user_predict_ids]

    return result

if __name__ == '__main__':
    eval(**args)
