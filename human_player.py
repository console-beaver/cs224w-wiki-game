#!/usr/bin/env python3

import os

if __name__ == '__main__':
    # first verify the dataset has been downloaded
    fail = False
    if os.path.isdir('dataset'):
        if not os.path.isfile(os.path.join('dataset', 'enwiki-2013.txt')):
            print('dataset/enwiki-2013.txt is missing, please run download_dataset.sh')
            fail = True
        if not os.path.isfile(os.path.join('dataset', 'enwiki-2013-names.csv')):
            print('dataset/enwiki-2013-names.csv is missing, please run download_dataset.sh')
            fail = True
    else:
        print('dataset directory is missing, please run download_dataset.sh')
        fail = True

    if not fail:
        # do stuff in here
        print('running demo (TODO)')


