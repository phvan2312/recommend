import pandas as pd
import numpy as np

data_path = './data_mini.txt'
data_sep  = ','

write_df_mini = False
df_mini_path  = './data_mini.txt'

create_user2id = True
user2id_path   = './data_user2id.txt'

split_test      = False
number_of_test  = 20000
test_path       = './data_mini_test.txt'
train_path      = './data_mini_train.txt'

if __name__ == '__main__':
    df = pd.read_csv(data_path,sep=data_sep)

    if write_df_mini:
        # get only 100k samples and write to file
        df_mini = df.head(n=200000)

        # get only users whose ids is less than 20k
        df_mini = df_mini[(df_mini['user_name_1'] <= 20000)&(df_mini['user_name_2'] <= 20000)]

        print ('number of samples: %d' % df_mini.shape[0])

        df_mini.to_csv(df_mini_path,index=False,columns=['user_name_1','user_name_2'])

    if split_test:
        duplicated_ids = df['user_name_1'].duplicated(keep='first')
        n_ids = duplicated_ids.shape[0]

        duplicated_ids[number_of_test:] = [False] * (len(duplicated_ids) - number_of_test)

        test_df  = df[duplicated_ids]
        train_df = df[duplicated_ids == False]

        train_df.to_csv(train_path, index=False, columns=['user_name_1', 'user_name_2'])
        test_df.to_csv(test_path, index=False, columns=['user_name_1', 'user_name_2'])

    if create_user2id:
        unique_user_name_1s = np.unique(df['user_name_1'].tolist())
        unique_user_name_2s = np.unique(df['user_name_2'].tolist())

        unique_user_names   = np.unique(np.concatenate([unique_user_name_1s,unique_user_name_2s]))
        user2id = {e:i for i,e in enumerate(unique_user_names)}

        print ('unique_user_name_1s: %d' % len(unique_user_name_1s))
        print ('unique_user_name_2s: %d' % len(unique_user_name_2s))
        print ('number of users: %d' % len(user2id))

        pd.DataFrame({'id':user2id.values(),'user_name':user2id.keys()})\
            .to_csv(user2id_path,index=False,columns=['id','user_name'])

