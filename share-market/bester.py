# -*- coding:utf-8 -*-
import pandas as pd
import json
from multiprocessing import Process
import os
import time


if __name__ == '__main__':
    traind = pd.read_csv('./data/log/train.csv', names=['low','high','close','ma5'])
    traind = pd.DataFrame( traind.mean(axis=1), columns=['mean'] )
    print(traind.sort_values(by=['mean']) )