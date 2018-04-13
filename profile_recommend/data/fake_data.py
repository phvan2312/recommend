import pandas as pd
import numpy as np

max_user = 200
n_user = 1000

out_path = './matrix.txt'
out_user2id_path = './user2id.txt'

if __name__ == '__main__':

    u1 = np.random.randint(0,max_user,n_user,dtype='int32')
    u2 = np.random.randint(0,max_user,n_user,dtype='int32')

    df = pd.DataFrame({'user_name_1': u1, 'user_name_2': u2})
    df = df[df['user_name_1'] != df['user_name_2']]

    print df.shape

    df.to_csv(out_path,sep=',',index=False)

    unique_users = np.arange(0,max_user)

    user2id_df = pd.DataFrame({'user_name':unique_users.tolist(),'id':unique_users.tolist()})
    user2id_df.to_csv(out_user2id_path,sep=',',index=False)


