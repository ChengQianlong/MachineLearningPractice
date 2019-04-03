"""
http://sofasofa.io/competition.php?id=7
运动员身价估计
xgboost 第一版
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import Imputer
from xgboost import plot_importance
import matplotlib.pyplot as plt
from datetime import date
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# load data
def loadData(path):
    data = pd.read_csv(path)
    return data
# deal missing value and feature selection

def getPlayAge(data):
    today = date(2019, 3, 24)
    data['birth_date'] = pd.to_datetime(data['birth_date'])
    data['age'] = (today - data['birth_date']).apply(lambda x: x.days) / 365
    # print(data.columns.values)
    # print(data.head(5))

def getBestPosition(data):
    positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']
    data['best_pos'] = data[positions].max(axis=1)
def getBMI(data):
    data['BMI'] = 10000. * data['weight_kg'] / (data['height_cm'] ** 2)
def getGK(data):
    data['is_gk'] = data['gk'] > 0

def featureSelect(data):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data.loc[:, ['club', 'league', 'height_cm', 'weight_kg', 'potential', 'BMI', 'pac', 'sho', 'dri', 'skill_moves', 'phy', 'international_reputation', 'age', 'best_pos']])
    x_new = imputer.transform(data.loc[:, ['club', 'league', 'height_cm', 'weight_kg', 'potential', 'BMI', 'pac', 'sho', 'dri', 'skill_moves', 'phy', 'international_reputation', 'age', 'best_pos']])
    data_num = len(x_new)
    XList = []
    yList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(x_new[row][0])
        tmp_list.append(x_new[row][1])
        tmp_list.append(x_new[row][2])
        tmp_list.append(x_new[row][3])
        tmp_list.append(x_new[row][4])
        tmp_list.append(x_new[row][5])
        tmp_list.append(x_new[row][6])
        tmp_list.append(x_new[row][7])
        tmp_list.append(x_new[row][8])
        tmp_list.append(x_new[row][9])
        tmp_list.append(x_new[row][10])
        tmp_list.append(x_new[row][11])
        tmp_list.append(x_new[row][12])
        tmp_list.append(x_new[row][13])
        XList.append(tmp_list)
        yList.append(data.iloc[row]['y'])

    F = f_regression(XList, yList)
    print(len(F))
    print(F)

def getTrainData(data_train):
    # deal miss val
    imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imputer.fit(data_train.loc[:, ['club', 'league', 'height_cm', 'weight_kg', 'potential', 'BMI', 'pac', 'sho', 'dri', 'skill_moves', 'phy', 'international_reputation', 'age', 'best_pos']])
    x_new = imputer.transform(data_train.loc[:, ['club', 'league', 'height_cm', 'weight_kg', 'potential', 'BMI', 'pac', 'sho', 'dri', 'skill_moves', 'phy', 'international_reputation', 'age', 'best_pos']])
    X_list = []
    Y_list = data_train.y.values
    for i in range(len(x_new)):
        tmp_list = []
        tmp_list.append(data_train.iloc[i]['club'])
        tmp_list.append(data_train.iloc[i]['league'])
        tmp_list.append(data_train.iloc[i]['potential'])
        tmp_list.append(data_train.iloc[i]['international_reputation'])
        tmp_list.append(data_train.iloc[i]['pac'])
        tmp_list.append(data_train.iloc[i]['sho'])
        tmp_list.append(data_train.iloc[i]['pas'])
        tmp_list.append(data_train.iloc[i]['dri'])
        tmp_list.append(data_train.iloc[i]['def'])
        tmp_list.append(data_train.iloc[i]['phy'])
        tmp_list.append(data_train.iloc[i]['skill_moves'])
        tmp_list.append(x_new[i][0])
        tmp_list.append(x_new[i][1])
        tmp_list.append(x_new[i][2])
        tmp_list.append(x_new[i][3])
        tmp_list.append(x_new[i][4])
        tmp_list.append(x_new[i][5])
        X_list.append(tmp_list)
    return X_list, Y_list


def getTestData(data_test):
    # deal miss val
    imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imputer.fit(data_test.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])
    x_new = imputer.transform(data_test.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])
    X_list = []
    for i in range(len(x_new)):
        tmp_list = []
        tmp_list.append(data_test.iloc[i]['club'])
        tmp_list.append(data_test.iloc[i]['league'])
        tmp_list.append(data_test.iloc[i]['potential'])
        tmp_list.append(data_test.iloc[i]['international_reputation'])
        tmp_list.append(data_test.iloc[i]['pac'])
        tmp_list.append(data_test.iloc[i]['sho'])
        tmp_list.append(data_test.iloc[i]['pas'])
        tmp_list.append(data_test.iloc[i]['dri'])
        tmp_list.append(data_test.iloc[i]['def'])
        tmp_list.append(data_test.iloc[i]['phy'])
        tmp_list.append(data_test.iloc[i]['skill_moves'])
        tmp_list.append(x_new[i][0])
        tmp_list.append(x_new[i][1])
        tmp_list.append(x_new[i][2])
        tmp_list.append(x_new[i][3])
        tmp_list.append(x_new[i][4])
        tmp_list.append(x_new[i][5])
        X_list.append(tmp_list)
    return X_list


def trainAndTest(X_train, Y_train, X_test):
    # XGboost fit model
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    model.fit(X_train, Y_train)
    # predict test data
    pred = model.predict(X_test)
    id_list = np.arange(10441, 17441)
    pred_arr = []
    for i in range(len(pred)):
        pred_arr.append([int(id_list[i]), pred[i]])
    np_pred = np.array(pred_arr)

    # write data to csv file
    pd_data = pd.DataFrame(np_pred, columns=['id', 'y'])
    pd_data.to_csv('D:/Study/data/data_sofasofa/submit.csv', index=None)
    print(pd_data)
    # show import feature
    plot_importance(model)
    plt.show()

def train(data, data_test, data_submit):
    data_test['pred'] = 0
    # 后面把身高，体重加上试一下
    cols_ngk = ['league', 'potential', 'BMI', 'pac', 'pas', 'sho', 'dri', 'def', 'skill_moves', 'penalties', 'vision',\
                'phy', 'finishing', 'ball_control', 'shot_power', 'international_reputation', 'age', 'best_pos']
    cols_gk = ['league', 'potential', 'BMI', 'pac', 'sho', 'dri', 'skill_moves', 'gk_diving', 'gk_handling', 'gk_kicking',\
               'gk_positioning', 'gk_reflexes', 'phy', 'international_reputation', 'age', 'best_pos']
    reg_ngk = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=180, silent=False, objective='reg:gamma')
    reg_ngk.fit(data[data['is_gk'] == False][cols_ngk], data[data['is_gk'] == False]['y'])
    preds = reg_ngk.predict(data_test[data_test['is_gk'] == False][cols_ngk])
    data_test.loc[data_test['is_gk'] == False, 'pred'] = preds

    reg_gk = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=180, silent=False, objective='reg:gamma')
    reg_gk.fit(data[data['is_gk'] == True][cols_gk], data[data['is_gk'] == True]['y'])
    preds = reg_gk.predict(data_test[data_test['is_gk'] == True][cols_gk])
    data_test.loc[data_test['is_gk'] == True, 'pred'] = preds

    # # 模型分数
    # kfold = KFold(n_splits=10)
    # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=180, silent=False, objective='reg:gamma')
    # cv_score = cross_val_score(model, data[:][cols], data[:]['y'], cv=kfold)
    # print("the kfold score is {}".format(cv_score))
    # print("the ngk_model score {}".format(reg_ngk.score(data[data['is_gk'] == False][cols], data[data['is_gk'] == False]['y'])))
    # print("the gk_model score {}".format(reg_gk.score(data[data['is_gk'] == True][cols], data[data['is_gk'] == True]['y'])))

    # 模型调参
    # cv_params = {'n_estimators': [140, 150, 160, 170, 180, 200, 210, 300]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    # optimized_GBM.fit(data[data['is_gk'] == False][cols], data[data['is_gk'] == False]['y'])
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    # 输出预测结果
    data_submit['y'] = np.array(data_test['pred'])
    data_submit.to_csv("D:/Study/data/data_sofasofa/submit.csv",index=False)

# 测试模型的性能
def model_test(X, Y, model):
    kfold = KFold(n_splits=10)
    cv_score = cross_val_score(model, X, Y, cv=kfold)

if __name__ == '__main__':
    trainpath = 'D:/Study/data/data_sofasofa/train.csv'
    testpath = 'D:/Study/data/data_sofasofa/test.csv'
    submitpath = 'D:/Study/data/data_sofasofa/sample_submit.csv'
    data_train = loadData(trainpath)
    data_test = loadData(testpath)
    data_submit = loadData(submitpath)
    # get age of player
    getPlayAge(data_train)
    getPlayAge(data_test)
    getBestPosition(data_train)
    getBestPosition(data_test)
    getBMI(data_train)
    getBMI(data_test)
    getGK(data_train)
    getGK(data_test)
    # print(data_train.columns.values)
    # print(data_train.head(5))
    # feature selection
    featureSelect(data_train)
    # train data
    X_train, Y_train = getTrainData(data_train)
    # test model

    # test data
    X_test = getTestData(data_test)
    # # train and test
    # trainAndTest(X_train, Y_train, X_test)

    train(data_train, data_test, data_submit)
    print('program running successful')
