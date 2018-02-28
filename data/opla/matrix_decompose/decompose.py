import sys
sys.path.append('..')

from opla_utils import matrix_decomposite
import pandas as pd

if __name__ == '__main__':
    rating_path = './../metadata/rating_matrix.csv'

    rating_datas = pd.read_csv(rating_path,sep=',')
    rating_matrix = rating_datas.as_matrix()

    U, V = matrix_decomposite(rating_matrix, k=200, n_iter=1000)

    U.astype('float32').tofile(open('U.csv.bin','w'))
    V.astype('float32').tofile(open('V.csv.bin','w'))