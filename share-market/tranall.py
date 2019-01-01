# -*- coding:utf-8 -*-
import pandas as pd
import json
from multiprocessing import Process
import os
import rnn_predict
import time

def run_calc(idx, total, id, name):
    rnn_predict.load(idx,total, id, name)

if __name__ == '__main__':
    traind = pd.read_csv('./data/log/train.csv', names=['id','low','high','close','ma5'])
    #print(traind.loc[traind.id == 600176])
    needcalc = []
    with open('./data/total.json', 'r') as f:
        marks = json.load(fp = f)
        for item in marks:
            id = int(item['id'])
            outdata = traind.loc[traind.id == id]
            if outdata.empty == True:
                calc = {'id':item['id'], 'name':item['name']}
                needcalc.append(calc)
    
    allsize = len(needcalc)
    idx = 1
    for item in needcalc:
        p = Process(target=run_calc, args=(idx, allsize, item['id'], item['name'],))
        p.start()
        p.join()
        time.sleep(20)
        idx = idx + 1