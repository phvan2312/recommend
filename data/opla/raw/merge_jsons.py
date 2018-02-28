
import sys
sys.path.append('..')

import json
from opla_utils import merge_multiple_json

if __name__ == '__main__':
    in_path  = './zipper-2018-01-23--08-01'
    out_path = './data.json'

    results = merge_multiple_json(path=in_path)

    json.dump(results,open(out_path,'w'))