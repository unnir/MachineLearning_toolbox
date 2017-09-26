import gc
import numpy as np


# Random hyperparameter search for xgboost 

import xgboost as xgb

xgb_pars = []
cnt = 0
for MCW in [1,2, 3]:
    for ETA in [0.04, 0.07, 0.06, 0.05]:
        for CS in [0.3, 0.23, 0.18]:
            for MDS in [1,2,4,5,7,8,9]:
                for MD in [3,4,5, 6, 7,12,14,18,20,25]:            
                    for GM in [0.1,0,0.05,0.001,0,0.05,0.01]:
                        for SS in [0.5, 0.6, 0.7, 0.8, 0.9]:
                            for LAMBDA in [0.5, 1., 1.5,2,3]:
                                for SEED in [1111,878]:
                                    xgb_pars.append({'min_child_weight': MCW, 
                                                     'eta': ETA, 
                                                     'colsample_bytree': CS,
                                                     'max_depth': MD,
                                                     'subsample': SS, 
                                                     'lambda': LAMBDA, 
                                                     'eval_metric': 'logloss',
                                                     'max_delta_step':MDS,
                                                     'silent': 1, 
                                                     'objective': 'binary:logistic',
                                                     'gamma':GM,
                                                     'seed':SEED})
d_xgb = xgb.DMatrix(X, y)

cnt = 0
base_score = 0.7

while cnt != 10:
    np.random.seed()
    xgb_par = np.random.choice(xgb_pars, 1)[0]
    cnt += 1
    
    es = xgb.cv(xgb_par, d_xgb, num_boost_round=2400, nfold=20, stratified=True,
             metrics={'logloss'}, seed = 1111, maximize=False, verbose_eval=50,
            callbacks=[#xgb.callback.print_evaluation(show_stdv=False),
                        xgb.callback.early_stop(5)])
    
    score = es['test-logloss-mean'].iloc[-1]
    print('Modeling logloss %.5f' % score)
    if score < base_score:
        base_score = score
        final_params_ = xgb_par
        print(final_params_)
    if cnt%100 == 0:
        gc.collect()
    print('\n CNT is %i'% cnt)
gc.collect()


# Random hyperparameter search for lightgbm

import lightgbm as lgb

lgb_train = lgb.Dataset(X,y)

lgb_pars = []
for MDL in [10, 20,  40, 60,  80, 100, 150,]:
    for MB in [100,200,300,400,500,600,700]:
        for MDB in [3, 4, 5, 6, 7, 8, 9]:
            for MD in [100,230,400,500, 150,306]:            
                for LR in [0.5, 0.01, 0.14,0.1,0.05]:
                    for SS in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        for LAMBDA in [0.5, 1., 0.2,0.1,0.05]:
                            for SEED in [111,1111,1212,23,36,878,888]:
                               # for BT in ['gbdt','dart']:
                                    lgb_pars.append({
                                                        #'boosting_type': BT,
                                                        'objective': 'multiclass',
                                                        'metric': 'multi_logloss',
                                                        'num_leaves': MD,
                                                        'min_data_in_leaf':MDL,
                                                        'min_data_in_bin':MDB,
                                                        'learning_rate': LR,
                                                        'feature_fraction': 0.9,
                                                        'bagging_fraction': SS,
                                                        'bagging_freq': 5,
                                                        'verbose': 0,
                                                        'num_class':5,
                                                        'max_bin':MB,
                                                        'seed':SEED,
                                                    })
gc.collect()
cnt = 0
base_score = 0.6
while cnt != 100:
    np.random.seed()
    lgb_par = np.random.choice(lgb_pars, 1)[0]
    cnt += 1
    a = lgb.cv(lgb_par, lgb_train, num_boost_round=1000, 
            folds=None, nfold=20, stratified=True, shuffle=True, 
            metrics='multi_logloss', fobj=None, feval=None, init_model=None, 
            feature_name='auto', categorical_feature='auto', 
            early_stopping_rounds=5, fpreproc=None, verbose_eval=None, 
            show_stdv=True, seed=1111, callbacks=None)
    score = a['multi_logloss-mean'][-1]
    print('Modeling logloss %.5f' % score)
    if score < base_score:
        base_score = score
        final_params_ll = lgb_par
        print(final_params_ll)
    if cnt%100 == 0:
        gc.collect()
    print('\n CNT is %i'% cnt)
gc.collect()
