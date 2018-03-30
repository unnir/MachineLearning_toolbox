from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time
import lightgbm as lg


cnt = 0

def objective(params):
    global cnt 
    params = {
        'num_leaves': int(params['num_leaves']),
        'max_bin':int(params['max_bin']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'learning_rate': '{:.3f}'.format(params['learning_rate']),
        'lambda_l2':int(params['lambda_l2']),

    }  
    cv_data = lg.cv(params, train,  num_boost_round=4000, nfold=5,  seed = 2332,
                    stratified = False,early_stopping_rounds=5, metrics='rmse')    
    score = cv_data['rmse-mean'][-1]
    
    # saving score to a file
    
    print("############### Score: {0}".format(score))
    print("############### Prms: ", params)
    print('..........................')
    with open("cv_lightgbm.txt", "a") as myfile:
        myfile.write(f'''
        ############### Score: {cnt}
        ############### Score: {score}
        ############### Prms:{params}
        \n
        ''')
    cnt += 1
    return {
        'loss': score,
        'status': STATUS_OK,
        'eval_time': time.time(),
        }

space = {
    'num_leaves': hp.quniform('num_leaves', 8, 614, 2),
    'max_bin': hp.quniform('max_bin', 8, 512, 1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1.0, 0.01),
    'learning_rate': hp.quniform('learning_rate', 0.005, 0.12, 0.001),
    'lambda_l2':hp.quniform('lambda_l2', 1, 88, 1),
}

trials = Trials()
best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials)
print(best)
