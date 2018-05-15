# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:47:49 2018

@author: lenovo
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

#训练
df1 = pd.read_csv('C:/Users/lenovo/Desktop/3.31前.csv',engine='python')
x,y = df1.drop(['无时间价格'],axis=1),df1[['无时间价格']]
# 随机抽取训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state = 10)

#随机森林
alg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=25,
           max_features=5, max_leaf_nodes=None,min_impurity_split=1e-07,
           min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=200,n_jobs=1, oob_score=False, random_state=10,
           verbose=0, warm_start=False)
alg.fit(x_train,y_train)


#导入预测集
c1 = pd.read_csv('C:/Users/lenovo/Desktop/4.1-4.30.csv',engine='python')
p_x,p_y = c1.drop(['无时间价格'],axis=1),c1[['无时间价格']]

#预测
result_rfp = alg.predict(p_x)
#测试结果
result_rfp = pd.DataFrame(result_rfp).astype(int)
result_rfp.columns=['模型价格']
result_rfp['成交价'] = p_y.reset_index(drop=True)
result_rfp['误差'] = result_rfp['成交价']-result_rfp['模型价格']
result_rfp['误差率'] = result_rfp['误差']/result_rfp['成交价']
gg5 = result_rfp[result_rfp.误差率.between(-0.05, 0.05)]
gg10 = result_rfp[result_rfp.误差率.between(-0.1, 0.1)]
print(gg5.误差率.size/result_rfp.成交价.size)
print(gg10.误差率.size/result_rfp.成交价.size)