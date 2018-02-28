import sys
sys.path.append('..')

from opla_utils import extract_warm_from_R, extract_cold_user_from_R
import pandas as pd

def remove_R_by_ids(R,ids):
    org_n_rows = R.shape[0]
    remain_row_ids = set(range(org_n_rows)) - set(ids)

    return R[list(remain_row_ids),:]

def dump_ids(data, path):
    with open(path,'w') as f:
        f.write('\n'.join(data))

def get_x_ids(R,x_col_id):
    ids = list(set(R[:,x_col_id]))
    ids_str = [str(id) for id in ids]
    return ids_str

def dump_R(R, data_path):
    u_ids, i_ids, rating = R.T.tolist()
    df = pd.DataFrame({'uid':u_ids,'iid':i_ids,'rating':rating},columns=['uid','iid','rating'],dtype='int32')
    df.to_csv(data_path,sep=',',header=False,index=False)

if __name__ == '__main__':
    rating_path = './../metadata/rating_matrix.csv'

    rating_datas = pd.read_csv(rating_path,sep=',')
    rating_matrix = rating_datas.as_matrix()

    # cold_start_user
    cold_user_matrix, selected_ids = extract_cold_user_from_R(rating_matrix, 0.1)
    rating_matrix = remove_R_by_ids(rating_matrix, selected_ids)
    dump_R(R=cold_user_matrix, data_path='./test_cold_user.csv')

    # warm start
    warm_matrix, selected_ids = extract_warm_from_R(rating_matrix, 0.3)
    rating_matrix = remove_R_by_ids(rating_matrix, selected_ids)
    dump_R(R=warm_matrix, data_path='./test_warm.csv')

    # remaining is train.csv
    dump_R(R=rating_matrix, data_path='./train.csv')