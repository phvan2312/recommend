import json
import numpy as np
#from pymf.nmf import NMF
import pandas as pd
import json

def matrix_decomposite(R,k=200,n_iter=200):
    """
    :param R: rating matrix
    :param k: latent dim
    :param n_iter: number of iterations for algorithm
    :return: U,V
    """
    n_users = np.max(R[:,0]) + 1
    n_items = np.max(R[:,1]) + 1

    new_R = np.zeros(shape=(n_users,n_items),dtype='float32')
    new_R[R[:,0].tolist(),R[:,1].tolist()] = R[:,2].astype('float32')

    mdl = NMF(new_R,num_bases=k)
    mdl.factorize(niter=n_iter)

    U,V = mdl.W, mdl.H.T

    return U,V

def merge_multiple_json(path):
    """
    Merge all json files which have same structure into one
    :return: list of json file
    """
    import glob

    result = {}
    file_names = glob.glob(path + '/*.json')

    for file_name in file_names:
        with open(file_name,'r') as f:
            data = json.load(f)
            if type(data) is dict: result[file_name.replace('.json','')] = data
            else: print 'fake_data in %s is invalid, pls check !!!' % file_name

    return result

def get_item_detail(item):
    key = 'skills'
    value = item.get(key, '')

    results = {'work':[],'skill':[]}

    for elems in value:
        if type(elems) is list:
            for elem in elems:
                work_pos = elem['work_pos']

                working_position = work_pos.get('summary', work_pos.get('studyType', ''))
                working_place = work_pos.get('company', work_pos.get('institution', ''))

                results['work'].append({'work':working_position,'place':working_place})
        else:
            skill = elems['skill']
            results['skill'].append({'skill':skill})

    return results

def get_profile_detail(profile):

    def get_basics_detail(_profile):
        required_key = 'basics'

        if required_key in _profile:
            result = _profile[required_key].get('label','')
        else:
            result = ''

        return required_key, result

    def get_work_detail(_profile):
        required_key = 'work'

        results = []
        for work_dct in _profile.get(required_key,[]):
            company = work_dct.get('company','')
            position = work_dct.get('summary','')

            results.append({'company':company,'position':position})

        results = json.dumps(results)

        return required_key, results

    def get_education_detail(_profile):
        required_key = 'education'

        results = []
        for education_dct in _profile.get(required_key,[]):
            institution = education_dct.get('institution','')
            study = education_dct.get('studyType','')

            results.append({'institution':institution,'study':study})

        results = json.dumps(results)

        return required_key, results

    def get_skills_detail(_profile):
        required_key = 'skills'

        results = [{'skill':e} for e in _profile.get(required_key,[])]
        results = json.dumps(results)

        return required_key, results

    features_func = [get_basics_detail, get_work_detail, get_education_detail, get_skills_detail]
    results = {}

    for feature_func in features_func:
        k,v = feature_func(_profile=profile)
        results[k] = v

    return results

def get_metadata(datas):
    """
    build metadata from raw datas. This method is designed for opla only. Require modifications if you use it for others.
    :param datas: data, type dictionary
    :return: utility matrix, item details, profile details
    """

    required_profile_key, required_item_key = 'basics', 'category'
    utility_matrix = []

    item_details, profile_details = {}, {}

    for _ , data in datas.items():
        profile_name, item_name = None, None

        # extracted profile data for user, because two user may have the same nick name
        # so we will concatenate user nick name and its profile link to form the unique one.
        if type(data) is dict and required_profile_key in data.keys() :
            profile_name = "%s|%s" % (data[required_profile_key].get('name',''),
                                      data[required_profile_key].get('profile',''))

            profile_details[profile_name] = get_profile_detail(data)
            print ('extracted data of profile: %s ...' % data[required_profile_key].get('name',''))

        # for item
        if type(data) is dict and required_item_key in data.keys():
            if hasattr(data[required_item_key],'items'):
                for k,v in data[required_item_key].items():

                    item_detail = get_item_detail(v)
                    print ('extracted data for category %s ...' % k)

                    #item_detail_to_str = json.dumps(item_detail) #" ; ".join(list(set(item_detail)))
                    if k in item_details:
                        item_details[k]['work'].append(item_detail['work'])
                        item_details[k]['skill'].append(item_detail['skill'])
                    else:
                        item_details[k] = {}
                        item_details[k]['work'] = [item_detail['work']]
                        item_details[k]['skill'] = [item_detail['skill']]

                    utility_matrix.append({
                        'profile': profile_name,
                        'item': k,
                        'rating':v['point']
                    })

    return utility_matrix, \
           {k:{'work':json.dumps(v['work']),'skill':json.dumps(v['skill'])} for k,v in item_details.items()}, \
           profile_details

def get_ids_rated_by_x(R,x_id,x_col_id):
    ids = np.where(R[:,x_col_id] == x_id)[0]

    return ids

def extract_warm_from_R(R,split_percent=0.1):
    """
    :param R: rating matrix
    :param split_percent: indicate number of samples to be extracted for each user
    :return: test warm rating matrix (R)
    """
    user_ids = np.unique(R[:, 0]).tolist()

    final_i_ids = []
    final_u_ids = []
    final_ratings = []
    final_ids = []

    # get from each users some items
    for user_id in user_ids:
        ids = get_ids_rated_by_x(R, x_id=user_id, x_col_id=0)
        rand_len = int(split_percent * ids.shape[0])
        select_ids = ids[np.random.permutation(ids.shape[0])][:rand_len]

        select_i_ids = R[select_ids,1]
        select_u_ids = R[select_ids,0]
        select_ratings = R[select_ids,2]

        if len(select_ids) > 0: final_ids.extend(select_ids.tolist())
        if len(select_i_ids) > 0: final_i_ids.extend(select_i_ids.tolist())
        if len(select_u_ids) > 0: final_u_ids.extend(select_u_ids.tolist())
        if len(select_ratings) > 0: final_ratings.extend(select_ratings.tolist())

    return pd.DataFrame(data={'uid': final_u_ids, 'iid': final_i_ids, 'rating': final_ratings},
                        columns=['uid', 'iid', 'rating']).as_matrix(), final_ids

def extract_cold_user_from_R(R, split_percent=0.1):
    """
    :param R: rating matrix
    :param split_percent: indicate number of samples to be extracted
    :return: test cold_user rating matrix (R)
    """

    user_ids = np.unique(R[:, 0])
    rand_len = int(split_percent * user_ids.shape[0])

    # get random
    selected_u_ids = user_ids[np.random.permutation(user_ids.shape[0])[:rand_len]]

    # get ids
    selected_ids = []
    for selected_u_id in selected_u_ids:
        ids = get_ids_rated_by_x(R, x_id=selected_u_id, x_col_id=0).tolist()
        selected_ids.extend(ids)

    select_i_ids = R[selected_ids,1]
    selected_ratings = R[selected_ids,2]
    selected_u_ids = R[selected_ids,0]

    # make sure everything ok
    assert selected_u_ids.shape[0] == select_i_ids.shape[0]
    assert selected_u_ids.shape[0] == selected_ratings.shape[0]

    return pd.DataFrame(data={'uid':selected_u_ids,'iid':select_i_ids,'rating':selected_ratings},
                        columns=['uid','iid','rating']).as_matrix(), selected_ids

def extract_cold_item_from_R(R, split_percent=0.1):
    """
    :param R: rating matrix
    :param split_percent: indicate number of samples to be extracted
    :return: test cold_item rating matrix (R)
    """

    R_T = R[:,[1,0,2]]
    result, selected_ids = extract_cold_user_from_R(R_T,split_percent)

    return result[:,[1,0,2]], selected_ids