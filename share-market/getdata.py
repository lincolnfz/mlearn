#!/usr/bin/python
# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
import pymysql
import numpy as np
from datetime import datetime

conn = pymysql.connect(host='127.0.0.1', user='root', password='123abc',
                             db='share_market',charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


def init_symbol():
    global conn
    cur=conn.cursor()
    hslist = ts.get_hs300s()

    values=[]
    for i in hslist.index:
        t =  list(hslist.loc[i, ['code','name','weight','date']] )
        tmp =  pd.Timestamp(t[3])
        t[3] = tmp.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
        t[2] = float(t[2])
        t = tuple(t)
        values.append(t)

    cur.executemany("insert into symbol(exchange_id, name, weight, last_updated_date) values (%s, %s,%s, %s)" ,values)
    conn.commit()
    cur.close()

def init_price():
    global conn
    cur = conn.cursor()
    cur.execute( "select * from symbol;" )
    result = cur.fetchall()
    for row in result:
         ticker = row['exchange_id']
         init_price_detail(ticker)

    cur.close()

def init_price_detail( ticker ):
    global conn
    cur = conn.cursor()
    hslist = ts.get_hist_data(ticker)
    values = []
    if hslist is not None:
        for i in hslist.index:
            t = list( hslist.loc[i] )
            t.insert(0, i)
            t.insert(0, ticker)
            t = tuple(t)
            values.append(t)

        cur.executemany("insert into daily_price(symbol_id, price_date, open_price, high_price, close_price,  low_price, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20) values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ", values)
        conn.commit()
    cur.close()
    print(values)
    pass

if __name__ == '__main__':
    #init_symbol()
    init_price()
    conn.close()