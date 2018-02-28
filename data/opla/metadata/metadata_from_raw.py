# this script extracts some useful information (called metadata) from raw data

import sys
sys.path.append('..')

import json
from opla_utils import get_metadata
import pandas as pd

if __name__ == '__main__':
    raw_data_path = "./../raw/data.json"
    raw_data = json.load(open(raw_data_path, 'r'))

    utility_matrix, item_details, profile_details = get_metadata(datas=raw_data)

    # builing mapping vocab to ids
    id2profile = {i:e for i,e in enumerate(profile_details.keys())}
    id2item = {i:e for i,e in enumerate(item_details.keys())}

    profile2id = {e:i for i,e in id2profile.items()}
    item2id = {e:i for i,e in id2item.items()}

    # convert all materials to ids
    for elem in utility_matrix:
        elem['profile'] = profile2id[elem['profile']]
        elem['item'] = item2id[elem['item']]

    item_details = {item2id[k]:";".join(v) for k,v in item_details.items()}
    profile_details = {profile2id[k]:v for k,v in profile_details.items()}

    df_item_details = pd.DataFrame(data=item_details.items(),columns=['id','desc'])
    df_item_details.to_csv('./item.csv',sep=',',index=False,encoding='utf-8')

    df_profile_details = pd.DataFrame.from_dict(profile_details, orient='index')
    df_profile_details['id'] = profile_details.keys()
    df_profile_details.to_csv('./profile.csv',sep=',',index=False,encoding='utf-8',columns=['id','basics','education',
                                                                                            'skills','work'])

    df_rating_matrix = pd.DataFrame(utility_matrix,columns=['profile','item','rating'])
    df_rating_matrix.to_csv('./rating_matrix.csv',sep=',',index=False)

    df_id2profile = pd.DataFrame(data=id2profile.items(), columns=['id','org_name'])
    df_id2item = pd.DataFrame(data=id2item.items(), columns=['id','org_name'])
    df_id2profile.to_csv('./id2profile.csv', sep=',',index=False,encoding='utf-8')
    df_id2item.to_csv('./id2item.csv', sep=',', index=False,encoding='utf-8')

