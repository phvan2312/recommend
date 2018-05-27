import numpy as np

def my_eval(target_matrix, predict_matrix):
    target_nonzezo = [sum(x) > 0 for x in target_matrix]
    target_nonzero = np.arange(0,len(target_matrix))[target_nonzezo]

    processed_target_matrix = target_matrix[target_nonzero]
    processed_predict_matrix = predict_matrix[target_nonzero]

    mul = processed_target_matrix * processed_predict_matrix

    return np.mean(np.sum(mul,axis=1) / np.sum(processed_target_matrix, axis=1))

def create_vocab(raw_datas):
    # raw_datas is a list of tuple (user_A,user_B), means user_A is following user_B .
    As = [raw_data[0] for raw_data in raw_datas]
    Bs = [raw_data[1] for raw_data in raw_datas]

    all = np.unique(np.hstack([As,Bs]))

    # return a mapping {original_user_id -> new_mapping_id}
    vocab = {u:i for i,u in enumerate(all)}
    return vocab

if __name__ == '__main__':
    pass