# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pymysql
import sys
from sqlalchemy import create_engine
import pymysql
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
from pylab import mpl
import subprocess
import json
import os


db_pass = '123abc'
 
def read_mysql_and_insert():
    
    try:
        conn = pymysql.connect(host='localhost',user='root',password='123abc',db='share_market',charset='utf8mb4')
    except pymysql.err.OperationalError as e:
        print('Error is '+str(e))
        sys.exit()
        
    try:
        engine = create_engine('mysql+pymysql://root:123abc@localhost:3306/share_market')
    except sqlalchemy.exc.OperationalError as e:
        print('Error is '+str(e))
        sys.exit()
    except sqlalchemy.exc.InternalError as e:
        print('Error is '+str(e))
        sys.exit()
        
    try:   
        sql = 'select exchange_id, name from symbol'
        df = pd.read_sql(sql, con=conn) 
    except pymysql.err.ProgrammingError as e:
        print('Error is '+str(e))
        sys.exit() 
 
    #print(df.head())
    #df.to_sql(name='sum_case_1',con=engine,if_exists='append',index=False)
    conn.close()
    #print('ok')
    return df

def read_mysql_and_insert_2():
    conn = pymysql.connect(host='localhost', user='root', password=db_pass,
                             db='share_market',charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
    sql = 'select exchange_id, name from symbol'
    df = pd.read_sql(sql, con=conn)
    #print(df.head())
    #df.to_sql(name='sum_case_1',con=engine,if_exists='append',index=False)
    conn.close()
    #print('ok')
    return df

def calcadfroot(exchage_id, name ):
    conn = pymysql.connect(host='localhost', user='root', password=db_pass,
                             db='share_market',charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

    sql = 'select * from daily_price where symbol_id = \'{id}\'  and price_date >= \'2016-01-01\'  order by price_date '.format(id=exchage_id)
    df = pd.read_sql(sql, con=conn)
    #print(df['close_price'])
    #print(df)
    if df.empty:
        return False

    series = pd.Series(data=df['close_price'])
    diff = series.diff(1)[1:] # dta[0] is nan
    #print(diff)
    #print(ts.adfuller(df['close_price']))
    #print(ts.adfuller(df['close_price'],1))
    adf_result = ts.adfuller(series)
    #print(  adf_result[0], adf_result[4]['5%'] )
    l0 = False
    if adf_result[0] < adf_result[4]['5%']:
        lab = '{},{},{}'.format(exchage_id, name, 'yes')
        l0 = True
    else:
        lab = '{},{},{}'.format(exchage_id, name,  'no')

    l1 = False
    diff_result = ts.adfuller(diff)
    if diff_result[0] < diff_result[4]['5%']:
        lab += ',yes'
        l1 = True
    else:
        lab += ',no'
    
    #print(exchage_id, name)
    #print(adf_result)
    plt.figure()
    plt.title(lab)
    plt.plot(series,'g-' )
    plt.plot(diff, 'r-')
    #plt.show()
    file = '{}.png'.format(name)
    plt.savefig(file)
    plt.close()
    conn.close()
    return l0 == False and l1 == True

def calcconit(ll):
    conint = []
    if os.path.isfile("pair.json"):
        with open('pair.json', 'r') as f:
            conint = json.load(f)
    else:
        for i in range(len(ll)):
            for j in range(len(ll)):
                if i != j and j > i:
                    pvalue = coint_2stocks(ll[i]['id'], ll[j]['id'])
                    if pvalue < 0.05:
                        item = {'stock1':ll[i]['id'], 'stcok2':ll[j]['id'], 'pvalue': pvalue }
                        conint.append(item)
                        #print(item, pvalue)
                    #break
            #break
        with open('pair.json', 'w') as f:
            json.dump(conint, f)
    df = pd.DataFrame.from_dict(conint)
    df = df.sort_values(by=['pvalue'])
    df = df.head(10)
    for i in range(0, len(df)):
        calc2stock(df.iloc[i,2], df.iloc[i,1])



def coint_2stocks(id1, id2):
    conn = pymysql.connect(host='localhost', user='root', password=db_pass,
                             db='share_market',charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

    sql = 'select * from daily_price where symbol_id = \'{id}\' order by price_date'.format(id=id1)
    df1 = pd.read_sql(sql, con=conn, index_col='price_date')
    if  df1.empty:
        return 1.0
    #series1 = pd.Series(data=df1['close_price'])
    d1 = df1.loc[:, 'close_price']
    

    sql = 'select * from daily_price where symbol_id = \'{id}\' order by price_date'.format(id=id2)
    df2 = pd.read_sql(sql, con=conn, index_col='price_date')
    if df2.empty:
        return 1.0

    #series2 = pd.Series(data=df2['close_price'])
    d2 = df2.loc[:, 'close_price']

    result = pd.concat([d1,d2], 1)
    result = result[~result.isin([np.nan, np.inf, -np.inf]).any(1)] #clean nan, inf -inf
    #print(result.iloc[:, 0])
    #print(result.iloc[:, 1])
    series1 = pd.Series(result.iloc[:, 0])
    series2 = pd.Series(result.iloc[:, 1])
    
    pvalue = 1
    try:
        coint_result = ts.coint( series1, series2 )
        pvalue = coint_result[1]
    except ValueError as identifier:
        pass
    #print(id1, id2, coint_result[1])
    
    conn.close()
    return pvalue

def calc2stock(id1, id2):
    conn = pymysql.connect(host='localhost', user='root', password=db_pass,
                             db='share_market',charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

    sql = 'select price_date, close_price as x from daily_price where symbol_id = \'{id}\' order by price_date'.format(id=id1)
    df1 = pd.read_sql(sql, con=conn, index_col='price_date')
    #series1 = pd.Series(data=df1['close_price'])
    d1 = df1.loc[:, 'x']

    sql = 'select price_date, close_price as y from daily_price where symbol_id = \'{id}\' order by price_date'.format(id=id2)
    df2 = pd.read_sql(sql, con=conn, index_col='price_date')
    #series2 = pd.Series(data=df2['close_price'])
    d2 = df2.loc[:, 'y']

    result = pd.concat([d1,d2], 1)
    result = result[~result.isin([np.nan, np.inf, -np.inf]).any(1)] #clean nan, inf -inf
    #print(result.iloc[:, 0])
    #print(result.iloc[:, 1])
    series1 = pd.Series(result.iloc[:, 0])
    series2 = pd.Series(result.iloc[:, 1])

    x = ts.add_constant(series1)
    result = (ts.OLS(series2, x)).fit()
    print(result.summary())
    #print(result.params['x'])
    diff =  zscore(series2 - result.params['x'] * series1)
    mean = np.mean(diff)
    std = np.std(diff)
    up = mean + std
    down = mean - std

    #mean_line = pd.Series(mean, index='price_date')
    #up_line = pd.Series(up, index='price_date')
    #down_line = pd.Series(down, index='price_date')
    #diff = pd.concat([diff, mean_line, up_line, down_line], axis=1)

    '''plt.figure()
    plt.plot(series1,'g-' )
    plt.plot(series2, 'r-')
    plt.show()
    plt.close()'''

    plt.figure()
    plt.plot(diff, 'b')
    plt.axhline(mean, color="black")
    plt.axhline(1.0, color="red", linestyle="--")
    plt.axhline(-1.0, color="green", linestyle="--")
    plt.legend(["z-score", "mean", "+1", "-1"])
    plt.show()
    plt.close()

def zscore(series):
    return (series - series.mean()) / np.std(series)

def trainStock(stockid, slice_size):
    slice_size += 1
    conn = pymysql.connect(host='localhost', user='root', password=db_pass,
                             db='share_market',charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

    sql = 'select  price_date, open_price, high_price, close_price, low_price, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20 from daily_price where symbol_id = \'{id}\' order by price_date'.format(id=stockid)
    df = pd.read_sql(sql, con=conn, index_col='price_date')
    while True:
        if df.empty:
            break
        #for idx in df.index:
        #    print(idx)
        total_slice = len(df.index) - slice_size + 1
        X = None
        Y = None
        for idx in  range(total_slice):
            x = df.iloc[idx:idx+slice_size-1]
            y = df.iloc[idx+slice_size-1]
            #x = x.reset_index()
            x_combin = x.iloc[0]
            #print(x_combin)
            #print(x_combin)
            for x_i in range(len(x.index)-1):
                x_item = x.iloc[x_i+1]
                x_combin = pd.concat( [x_combin, x_item], axis=0 )
            #x_combin = x_combin.dorp(['id', 'symbol_id', 'created_date', 'last_updated_date', 'adj_close_price', ''], asix=1)
            #print(x_combin)
            #break
            #X.append(x_combin)
            #Y.append(y)
            if X is None:
                X = x_combin
            else:
                X = pd.concat([X, x_combin], axis=1)

            if Y is None:
                Y = y
            else:
                Y = pd.concat([Y, y], axis=1 )

        #print(len(X.index))
        #print(len(Y.index))
        #print(X.T)
        #print(Y.T)
        break
    conn.close()
    return X.T, Y.T

if __name__ == '__main__':
    '''# Test 1
    # 定义数据
    dates = pd.date_range('20170101', periods = 6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index = dates, columns = ['A', 'B', 'C', 'D'])
    # 假设缺少数据
    df.iloc[1, 1] = np.nan
    df.iloc[2, 2] = np.nan
    print(df)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)] #clean nan, inf -inf
    print(df)
    exit'''

    ll = []
    if os.path.isfile("data.json"):
        with open('data.json', 'r') as f:
            ll = json.load(f)
    else:
        df = read_mysql_and_insert_2()
        for idx in df.index:
            row = df.loc[idx, ['exchange_id','name']]
            if row is None:
                continue
            ret = calcadfroot(row[0], row[1])
            if ret == True:
                item = { 'id': row[0], 'name': row[1] }
                ll.append(item)

        with open('data.json', 'w') as f:
            json.dump(ll, f)
    #calcconit(ll)
    X, Y = trainStock('601857', 5)
    print(X)
    print(Y)
    #print(df.head())

