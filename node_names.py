#!/usr/bin/env python3

import sys
import os
import pickle

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('number of subsamples was not specified')
        exit(1)
    n = int(sys.argv[1])
    if not os.path.exists(f'pickles/name_to_id_top_{n}.pkl'):
        print('this subsampled dataset was not generated yet, please generate it')
        exit(1)
    with open(f'pickles/name_to_id_top_{n}.pkl', 'rb') as f:
        name_to_id = pickle.load(f)
    print('this dataset\'s keys (node names) are:')
    print(name_to_id.keys())
