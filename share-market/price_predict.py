# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import pymysql
import sys
from sqlalchemy import create_engine
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
from pylab import mpl
import subprocess
import json
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import lightgbm as lgb
import tensorflow as tf

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

def gusweight(alpha, dist):
    return np.exp(-alpha * (np.power(dist, 2)))

def getDataStock(stockid, begin_date, slice_group, predict=1):
    slice_size = slice_group + predict
    conn = pymysql.connect(host='localhost', user='root', password=db_pass,
                             db='share_market',charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

    sql = 'select  price_date, open_price, high_price, close_price, low_price, volume, price_change, p_change, ma5, ma10, ma20, v_ma5, v_ma10, v_ma20 from daily_price where symbol_id = \'{id}\' and price_date > \'{date}\' order by price_date'.format(id=stockid, date=begin_date)
    df = pd.read_sql(sql, con=conn, index_col='price_date')
    while True:
        if df.empty:
            break
        #for idx in df.index:
        #    print(idx)
        total_slice = len(df.index) - slice_size + 1
        X = None
        Y = None
        width = 0
        df = (df - df.mean()) / (df.std()+0.0001)

        for idx in  range(total_slice):
            x = df.iloc[idx:idx+slice_group]
            y = df.iloc[idx+slice_group: idx+slice_group+predict]
            #x = x.reset_index()
            x_combin = x.iloc[0]
            width = x_combin.shape[0]
            #print(x_combin)
            #print(x_combin)
            for x_i in range(len(x.index)-1):
                x_item = x.iloc[x_i+1]
                #print(weight_pod, gusweight(0.1, weight_pod))
                x_combin = pd.concat( [x_combin, x_item], axis=0 )
            #x_combin = x_combin.dorp(['id', 'symbol_id', 'created_date', 'last_updated_date', 'adj_close_price', ''], asix=1)
            '''if y_ma5 - last_ma5 <= 0:
                y_value = 0
            else:
                y_value = 1'''
            y_item = []
            for y_i in y.index:
                val = (y.loc[y_i, 'high_price'] + y.loc[y_i, 'low_price']) / 2.0
                y_item.append( val )
            y_item = np.array(y_item)
            y_item = pd.DataFrame(data=y_item, columns=['qu'])
            #print(y_item)
            #break
            #X.append(x_combin)
            #Y.append(y)
            if X is None:
                X = x_combin
            else:
                X = pd.concat([X, x_combin], axis=1)

            if Y is None:
                Y = y_item
            else:
                Y = pd.concat([Y, y_item], axis=1 )
            

        break
    Y = Y.T
    X = X.T
    conn.close()
    
    #print(X.shape,Y.shape)
    # Build a forest and compute the feature importances
    return X,Y,width,slice_group

def forest_train(X, Y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=0)

    forest.fit(X, Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #print(X_new)
    #return X.T, Y.T
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

def rand_forest(X, Y):
    clf = RandomForestClassifier(n_estimators=500,
        min_samples_split=2, random_state=0)
    clf.fit(X, Y)
    #print(clf.score(test_X, test_Y))
    #clf.predict()
    #scores =  cross_val_score(clf, X, Y)
    #print(scores.mean())
    '''importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))'''
    return clf

def granient_classification(X, Y):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0).fit(X, Y)
    clf.fit(X, Y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    remove_col = indices[40:]
    X = np.array(X)
    X = np.delete(X, remove_col, axis=1)
    #print(X.shape)
    clf.fit(X, Y)
    return clf, remove_col 

def calcstock_act(id, name):
    X, Y = getDataStock(id,'2015-08-01', 5)
    if X is None:
        return 0.0
        
    test_num = 30
    data_len = len(X.index)

    g = (test_num+1-x for x in range(1, test_num+1))
    scores = []
    for nn in g:
        headnum = data_len - nn
        train_x = X.head(headnum)
        train_y = Y[:headnum]
        #train_x = (train_x - train_x.mean()) / (train_x.std())
        clf, remove_col = granient_classification(X=train_x, Y=train_y)

        test_x =  np.array(X.iloc[headnum]).reshape(1,-1)
        test_y = np.array(Y[headnum]).reshape(1)
        test_x = np.array(test_x)
        test_x = np.delete(test_x, remove_col, axis=1)
        ss = clf.score(test_x, test_y)
        scores.append(ss)
        #print(clf.predict(test_x), Y[headnum], ss, clf.predict_proba(test_x))
        #print(test_x.shape)
        #print(test_y.shape)
        #break
    scores = np.array(scores)
    avg = scores.mean()
    print('%s, avg: %f'% (name, avg) )
    return avg

def svc_cl(X, Y):
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, Y)
    return clf


def svc_classifation(id, name):
    X, Y = getDataStock(id,'2015-08-01', 5)
    test_num = 30
    data_len = len(X.index)

    g = (test_num+1-x for x in range(1, test_num+1))
    scores = []
    for nn in g:
        headnum = data_len - nn
        train_x = X.head(headnum)
        train_y = Y[:headnum]
        #train_x = (train_x - train_x.mean()) / (train_x.std())
        clf = svc_cl(X=train_x, Y=train_y)
        test_x =  np.array(X.iloc[headnum]).reshape(1,-1)
        test_y = np.array(Y[headnum]).reshape(1)
        ss = clf.score(test_x, test_y)
        scores.append(ss)
        #print(clf.predict(test_x), Y[headnum], ss, clf.predict_proba(test_x))
        #print(test_x.shape)
        #print(test_y.shape)
        #break
    scores = np.array(scores)
    avg = scores.mean()
    print('%s, avg: %f'% (name, avg) )
    return avg

def lightgdm_classifation(id, name):
    X, Y = getDataStock(id,'2015-08-01', 5)
    test_num = 10
    data_len = len(X.index)

    g = (test_num+1-x for x in range(1, test_num+1))
    scores = []
    for nn in g:
        headnum = data_len - nn
        train_x = X.head(headnum)
        train_y = Y[:headnum]
        train_data = lgb.Dataset(train_x, label=train_y )
        #print(train_data)
        param = {'num_leaves':20,  'objective':'binary'}
        param['metric'] = 'auc'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
        #print(bst.feature_importance())
        test_x =  np.array(X.iloc[headnum]).reshape(1,-1)
        test_y = np.array(Y[headnum]).reshape(1)
        ypred = bst.predict(test_x, num_iteration=bst.best_iteration)
        #print(test_y, ypred)
        y_out = 0
        if ypred > 0.5 :
            y_out = 1
        if y_out == test_y:
            scores.append(1)
        else:
            scores.append(0)
        
    print(np.array( scores ).mean())
        #
        
        
    #scores = np.array(scores)
    #avg = scores.mean()
    #print('%s, avg: %f'% (name, avg) )
    #return avg
    return 0.0

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

if __name__ == '__main__':
    if os.path.exists('./data') == False:
        os.makedirs('./data')
    symbolid = read_mysql_and_insert_2()
    data = []
    datalen = []
    nu = 0
    predict = 3
    for idx in symbolid.index:
        row = symbolid.loc[idx, ['exchange_id','name']]
        X, Y, width, height = getDataStock(row['exchange_id'], '2000-01-01', 30, predict)
        X = np.array(X)
        Y = np.array(Y)
        X[0].tostring
        writer = tf.python_io.TFRecordWriter('./data/%s.tfrecord' % row['exchange_id'])
        for idx in range(X.shape[0]):
            features = {}
            features['X'] = _bytes_feature(X[idx].tostring())
            features['Y'] = _bytes_feature(Y[idx].tostring())
            features['x_row'] = _int64_feature(height)
            features['x_col'] = _int64_feature(width)
            features['y_row'] = _int64_feature(1)
            features['y_col'] = _int64_feature(predict)
            tf_features = tf.train.Features(feature= features)
            example = tf.train.Example(features = tf_features)
            writer.write(example.SerializeToString())
        writer.close()
        print( '%d - %s done' % (nu, row['name']) )
        nu = nu + 1
        break