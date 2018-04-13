import numpy as np
import pandas as pd

def my_eval(target_matrix, predict_matrix):
    target_nonzezo = [sum(x) > 0 for x in target_matrix]
    target_nonzero = np.arange(0,len(target_matrix))[target_nonzezo]

    processed_target_matrix = target_matrix[target_nonzero]
    processed_predict_matrix = predict_matrix[target_nonzero]

    mul = processed_target_matrix * processed_predict_matrix

    return np.mean(np.sum(mul,axis=1) / np.sum(processed_target_matrix, axis=1))

if __name__ == '__main__':
    pass