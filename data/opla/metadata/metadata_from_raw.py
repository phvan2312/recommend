"""
Usage       : assume that we're in metadata folder
Test Date   : 07/05/2018
python metadata_from_raw.py --raw_data_path=./../raw/data.json --save_item_path=./out/item.csv --save_profile_path=./out/profile.csv
--save_rating_matrix_path=./out/rating_matrix.csv --save_id2profile_path=./out/id2profile.csv --save_id2item_path=./out/id2item.csv
"""



import sys
sys.path.append('..')

import json
from opla_utils import get_metadata
import pandas as pd
import argparse

parser  = argparse.ArgumentParser()

parser.add_argument('--raw_data_path',help='path for loading data of user (JSON type)',dest='raw_data_path',type=str)
parser.add_argument('--save_item_path',help='path for storing item information',dest='save_item_path',type=str)
parser.add_argument('--save_profile_path',help='path for storing profile path',dest='save_profile_path',type=str)
parser.add_argument('--save_rating_matrix_path',help='path for storing rating_matrix path',dest='save_rating_matrix_path',type=str)
parser.add_argument('--save_id2profile_path',help='path for storing id2profile path',dest='save_id2profile_path',type=str)
parser.add_argument('--save_id2item_path',help='path for storing id2item path',dest='save_id2item_path',type=str)

args    = vars(parser.parse_args())

raw_data_path           = args['raw_data_path'] #"./../raw/fake_data.json"
save_item_path          = args['save_item_path'] #'./item.csv'
save_profile_path       = args['save_profile_path'] #'./profile.csv'
save_rating_matrix_path = args['save_rating_matrix_path'] #'./rating_matrix.csv'
save_id2profile_path    = args['save_id2profile_path'] #'./id2profile.csv'
save_id2item_path       = args['save_id2item_path'] #'./id2item.csv'

def get_unique(datas,is_dumped = False):
    if is_dumped: datas = json.loads(datas)

    # "datas is a list of dict"
    dump_datas = set([json.dumps(data) for data in datas])

    return [json.loads(data) for data in dump_datas]

def update(datas):
    # datas is type of dict <key,values>, its key is the combination
    # of <name,link> and values is information.

    # This method will load fake_data from many global paths then update using datas variable.

    utility_matrix, item_details, profile_details = get_metadata(datas=datas)
    print ('extracted data sucessfully ...')

    # check if having new id {item,profile}.
    old_id2profile = pd.read_csv(save_id2profile_path, sep=',', encoding='utf-8')
    old_id2item = pd.read_csv(save_id2item_path, sep=',', encoding='utf-8')

    profile2id = {v:k for k,v in zip(old_id2profile['id'].tolist(),old_id2profile['org_name'].tolist())}
    item2id = {v:k for k,v in zip(old_id2item['id'].tolist(),old_id2item['org_name'].tolist())}

    print ('loaded old mapping vocabulary, with %d number of users and %d number of items ...' %
           (len(profile2id), len(item2id)))

    # update profile2id
    for item_name in item_details.keys():
        if item_name not in item2id:
            item2id[item_name] = len(item2id)

    # update profile2id
    for profile_name in profile_details.keys():
        if profile_name not in profile2id:
            profile2id[profile_name] = len(profile2id)
    print ('updated mapping vocabulary, with %d number of users and %d number of items ...' %
           (len(profile2id), len(item2id)))

    # convert all materials to ids
    for elem in utility_matrix:
        elem['profile'] = profile2id[elem['profile']]
        elem['item'] = item2id[elem['item']]

    old_item_details = pd.read_csv(save_item_path, sep=',', encoding='utf-8')
    old_profile_details = pd.read_csv(save_profile_path, sep=',', encoding='utf-8')

    # update item details
    for item_name, item_detail in item_details.items():
        item_id = item2id[item_name]

        # item_id not in old_item_details, add new
        if item_id not in old_item_details['id']:
            candidate = {
                'id': item_id,
                'skill':item_detail['skill'],
                'work':item_detail['work']
            }

            old_item_details = old_item_details.append(candidate,ignore_index=True)

        # item_id has already in old_item_details, append
        else:
            i_loc = old_item_details.index[old_item_details['id'] == item_id]
            print ('df_item_detail iloc: ', i_loc)
            cur_row = old_item_details.loc[i_loc]
            cur_skill, cur_work = json.loads(cur_row['skill'].values[0]), json.loads(cur_row['work'].values[0])

            cur_skill += json.loads(item_detail['skill'])
            cur_work  += json.loads(item_detail['work'])

            old_item_details.loc[i_loc, 'skill'] = json.dumps(get_unique(cur_skill))
            old_item_details.loc[i_loc, 'work']  = json.dumps(get_unique(cur_work))
    print ('df_item_deitals shape: ', old_item_details.shape)

    # update profile details
    for profile_name, profile_detail in profile_details.items():
        profile_id = profile2id[profile_name]

        # profile_id not in old_item_details, add new
        if profile_id not in old_profile_details['id']:
            candidate = {
                'id': profile_id,
                'basics': profile_detail['basics'],
                'education': profile_detail['education'],
                'skills': profile_detail['skills'],
                'work':profile_detail['work']
            }

            old_profile_details = old_profile_details.append(candidate,ignore_index=True)

        # profile_id has already in old_item_details, append
        else:
            i_loc = old_profile_details.index[old_profile_details['id'] == profile_id]
            print ('df_user_details iloc: ',i_loc)
            cur_row = old_profile_details.loc[i_loc]
            cur_basics, cur_education, cur_skills, cur_work = cur_row['basics'].values[0], json.loads(cur_row['education'].values[0]),\
                                                              json.loads(cur_row['skills'].values[0]),json.loads(cur_row['work'].values[0])

            cur_basics = profile_detail['basics']
            cur_education += json.loads(profile_detail['education'])
            cur_skills += json.loads(profile_detail['skills'])
            cur_work += json.loads(profile_detail['work'])

            old_profile_details.loc[i_loc, 'basics'] = cur_basics
            old_profile_details.loc[i_loc, 'education'] = json.dumps(get_unique(cur_education))
            old_profile_details.loc[i_loc, 'skills'] = json.dumps(get_unique(cur_skills))
            old_profile_details.loc[i_loc, 'work'] = json.dumps(get_unique(cur_work))

    print ('df_profile_details shape: ', old_profile_details.shape)

    # update rating_matrix
    old_rating_matrix = pd.read_csv(save_rating_matrix_path, sep=',')
    old_rating_matrix = old_rating_matrix.append(utility_matrix,ignore_index=True).drop_duplicates()
    print ('df_rating_matrix shape: ', old_rating_matrix.shape)

    df_id2profile = pd.DataFrame(data=profile2id.items(), columns=['org_name','id'])[['id','org_name']]
    df_id2item = pd.DataFrame(data=item2id.items(), columns=['org_name','id'])[['id','org_name']]

    print ('df_id2profile shape: ', df_id2profile.shape)
    print ('df_id2item shape: ', df_id2item.shape)

    save(df_id2profile,df_id2item,old_item_details,old_profile_details,old_rating_matrix)
    print 'saved ...'

def save(df_id2profile, df_id2item, df_item_details, df_profile_details, df_rating_matrix):
    df_id2profile.to_csv(save_id2profile_path, sep=',', index=False, encoding='utf-8')
    print ('saved id2profile to %s ...' % save_id2profile_path)

    df_id2item.to_csv(save_id2item_path, sep=',', index=False, encoding='utf-8')
    print ('saved id2item to %s ...' % save_id2item_path)

    df_item_details.to_csv(save_item_path, sep=',', index=False, encoding='utf-8')
    print ('saved item_details to %s ...' % save_item_path)

    df_profile_details.to_csv(save_profile_path, sep=',', index=False, encoding='utf-8',
                              columns=['id', 'basics', 'education', 'skills', 'work'])
    print ('saved profile_details to %s ...' % save_profile_path)

    df_rating_matrix.to_csv(save_rating_matrix_path, sep=',', index=False)
    print ('saved rating_matrix to %s ...' % save_rating_matrix_path)

def main(datas):
    utility_matrix, item_details, profile_details = get_metadata(datas=datas)
    print ('extracted data sucessfully ...')

    # builing mapping vocab to ids
    id2profile = {i: e for i, e in enumerate(profile_details.keys())}
    id2item = {i: e for i, e in enumerate(item_details.keys())}

    profile2id = {e: i for i, e in id2profile.items()}
    item2id = {e: i for i, e in id2item.items()}
    print ('created mapping vocabulary, with %d number of users and %d number of items ...' %
           (len(id2profile),len(id2item)))

    # convert all materials to ids
    for elem in utility_matrix:
        elem['profile'] = profile2id[elem['profile']]
        elem['item'] = item2id[elem['item']]

    # create item_details pandas
    df_item_details = pd.DataFrame()
    df_item_details['id'] = [item2id[k] for k in item_details.keys()]
    df_item_details['skill'] = [v['skill'] for v in item_details.values()]
    df_item_details['work'] = [v['work'] for v in item_details.values()]
    print ('df_item_deitals shape: ', df_item_details.shape)

    # create profile_details pandas
    df_profile_details = pd.DataFrame()
    df_profile_details['id'] = [profile2id[k] for k in profile_details.keys()]
    df_profile_details['basics'] = [v['basics'] for v in profile_details.values()]
    df_profile_details['education'] = [v['education'] for v in profile_details.values()]
    df_profile_details['skills'] = [v['skills'] for v in profile_details.values()]
    df_profile_details['work'] = [v['work'] for v in profile_details.values()]
    print ('df_profile_details shape: ', df_profile_details.shape)

    # create rating matrix pandas
    df_rating_matrix = pd.DataFrame(utility_matrix, columns=['profile', 'item', 'rating'])
    print ('df_rating_matrix shape: ', df_rating_matrix.shape)

    # create other necessary materials
    df_id2profile = pd.DataFrame(data=id2profile.items(), columns=['id', 'org_name'])
    df_id2item = pd.DataFrame(data=id2item.items(), columns=['id', 'org_name'])
    print ('df_id2profile shape: ', df_id2profile.shape)
    print ('df_id2item shape: ', df_id2item.shape)

    # save
    save(df_id2profile, df_id2item, df_item_details, df_profile_details, df_rating_matrix)
    print 'saved ...'

if __name__ == '__main__':
    raw_data = json.load(open(raw_data_path, 'r'))
    from_scratch_max = 5

    from_scratch_data, update_data = {}, {}
    for i,(k,v) in enumerate(raw_data.items()):
        if i > from_scratch_max: update_data[k] = v
        else: from_scratch_data[k] = v

    # create from scratch
    #raw_data = json.load(open(raw_data_path, 'r'))
    main(from_scratch_data)

    # # update
    # updated_data = json.load(open(raw_data_path, 'r'))
    update(update_data)



