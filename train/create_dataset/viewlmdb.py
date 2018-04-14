# coding: utf-8

import lmdb  # install lmdb by "pip install lmdb"

outputPath = '../data/lmdb/train'
env = lmdb.open(outputPath)
txn = env.begin(write=False)
for key, value in txn.cursor():
    print(key, value)

env.close()
