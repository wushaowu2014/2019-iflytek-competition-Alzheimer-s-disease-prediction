# -*- coding: utf-8 -*-
"""
@author: wushaowu

任务：根据老人的音频和文本（也就是说话内容），预测该老人属于哪种情况（CTRL：健康；MCI：轻度认知障碍
；AD：可能是阿尔茨海默综合症或其他种类的痴呆症）
注：官方给的训练数据179例；测试数据27例；训练集中只包含CTRL和AD两种。

本baseline 线上分数77+
"""
import os
import pandas as pd
import numpy as np
import time
from tqdm import *
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,recall_score,f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from collections import Counter
def one_hot_col(col):
    '''标签编码'''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(col)
    return lbl
def lgb_model(new_train,y,new_test):
    params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_leaves': 1000,
    'verbose': -1,
    'max_depth': -1,
  #  'reg_alpha':2.2,
  #  'reg_lambda':1.4,
    'seed':42,
    }
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof_lgb=np.zeros(new_train.shape[0]) ##用于存放训练集概率，由每折验证集所得
    prediction_lgb=np.zeros(new_test.shape[0])  ##用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame() ##存放特征重要性，此处不考虑
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        dtrain = lgb.Dataset(new_train.loc[tr],y[tr])
        dvalid = lgb.Dataset(new_train.loc[va],y[va],reference=dtrain)
        ##训练：
        bst = lgb.train(params, dtrain, num_boost_round=30000, valid_sets=dvalid, verbose_eval=400,early_stopping_rounds=200)
        ##预测验证集：
        oof_lgb[va] += bst.predict(new_train.loc[va], num_iteration=bst.best_iteration)
        ##预测测试集：
        prediction_lgb += bst.predict(new_test, num_iteration=bst.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(new_train.columns)
        fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
    
    print('the roc_auc_score for train:',roc_auc_score(y,oof_lgb)) ##线下auc评分
    print('the recall_score for train:',recall_score(y,[1 if i>0.5 else 0 for i in oof_lgb], average='macro'))
    print('the f1_score for train:',f1_score(y,[1 if i>0.5 else 0 for i in oof_lgb],average='weighted'))

    prediction_lgb/=5
    return oof_lgb,prediction_lgb,feature_importance_df

##读取数据：
preliminary_list_test=pd.read_csv("data/1_preliminary_list_test.csv") #大小（27,2）
preliminary_list_train=pd.read_csv("data/1_preliminary_list_train.csv") #大小（179,2）
egemaps_pre=pd.read_csv("data/egemaps_pre.csv") #大小（206,89）
print('训练集标签分布：\n',preliminary_list_train['label'].value_counts())

#==================================================================================
##读取转写文本，即tsv文件夹下的文件，每一个文件都是对应一个人的，且每个文件行数不一定相等。
tsv_path_lists=os.listdir('data/tsv/') #大小：206
tsv_feats=[] ##用于存放tsv特征
for path in tqdm(tsv_path_lists): ##遍历每个文件，提取特征
    z=pd.read_csv('data/tsv/'+path,sep='\t')
    ##说一句话所用时长：
    z['end_time-start_time']=z['end_time']-z['start_time']
    tsv_feats.append([path[:-4],\
                      z['end_time-start_time'].mean(),\
                      z['end_time-start_time'].min(),\
                      z['end_time-start_time'].max(),\
                      z['end_time-start_time'].std(),\
                      z['end_time-start_time'].median(),\
                      z['end_time-start_time'].skew(),\
                      z.shape[0]])
tsv_feats=pd.DataFrame(tsv_feats)
tsv_feats.columns=['uuid']+['tsv_feats{}'.format(i) for i in range(tsv_feats.shape[1]-1)]
#====================================================================================
##读取帧级别的Low-level descriptors (LLD)特征，即egemaps文件夹下的文件，每一个文件都是对应一个人的，且每个文件行数不一定相等。
##字段含义参考文献2
egemaps_path_lists=os.listdir('data/egemaps/') #大小：206
egemaps_feats=[] ##用于存放egemaps特征
for path in tqdm(egemaps_path_lists): ##遍历每个文件，提取特征
    z=pd.read_csv('data/egemaps/'+path,sep=';')
    z=z.drop(['name'],axis=1)
    egemaps_feats.append([path[:-4]]+\
                         list(z.mean(axis=0))+\
                         list(z.std(axis=0))+\
                         list(z.min(axis=0))+\
                         list(z.median(axis=0))) ##这里只求每列的平均值
egemaps_feats=pd.DataFrame(egemaps_feats)
egemaps_feats.columns=['uuid']+['egemaps_feats{}'.format(i) for i in range(egemaps_feats.shape[1]-1)]

#===========================分割线=============================================
##结合特征 ：
preliminary_list_train=preliminary_list_train.merge(egemaps_pre,how='left',on=['uuid'])
preliminary_list_train=preliminary_list_train.merge(tsv_feats,how='left',on=['uuid'])
preliminary_list_train=preliminary_list_train.merge(egemaps_feats,how='left',on=['uuid'])

preliminary_list_test=preliminary_list_test.merge(egemaps_pre,how='left',on=['uuid'])
preliminary_list_test=preliminary_list_test.merge(tsv_feats,how='left',on=['uuid'])
preliminary_list_test=preliminary_list_test.merge(egemaps_feats,how='left',on=['uuid'])

##标签映射：
label_dict={'CTRL':0,'AD':1}
preliminary_list_train['label']=preliminary_list_train['label'].map(label_dict)

#===========================分割线=============================================
##模型训练预测
oof_lgb,prediction_lgb,feature_importance_df=\
      lgb_model(preliminary_list_train.drop(['uuid','label'],axis=1),\
                preliminary_list_train['label'],\
                preliminary_list_test.drop(['uuid','label'],axis=1))

##保存结果：
result=[1 if i>0.5 else 0 for i in prediction_lgb]
preliminary_list_test['label']=result
print(preliminary_list_test['label'].value_counts())
preliminary_list_test['label']=preliminary_list_test['label'].apply(lambda x:'AD' if x==1 else 'CTRL')
preliminary_list_test[['uuid','label']].to_csv('submit.csv',index=None)