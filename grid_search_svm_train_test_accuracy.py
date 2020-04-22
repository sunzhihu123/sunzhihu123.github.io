# -*- coding: utf-8 -*-
"""
noting: u need rectify the path of training data and testing dada
note: each practice the result are different
"""
# from sklearn.model_selection import cross_val_score
# out precision of prediction
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np
import datetime
# 文件路径操作
import os
# svm and best parameter select using grid search method
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# scale the data to 0-1
from sklearn import preprocessing

def grid_find(train_data_x, train_data_y):
    # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
    # 10 is often helpful. Using a basis of 2, a finer.tuning can be achieved but at a much higher cost.
    C_range = np.logspace(-5, 9, 8, base=2)  # logspace(a,b,N),base默认=10，把10的a次方到10的b次方区间分成N份
    gamma_range = np.logspace(-15, 3, 10, base=2)
    print(C_range)
    print(gamma_range)
    parameters = {'kernel': ('linear', 'rbf'), 'C': C_range, 'gamma': gamma_range}
    svr = svm.SVC()
    # 更多关于网格搜索法
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # n_jobs表示并行运算量
    clf = GridSearchCV(svr, parameters, cv=5, n_jobs=4)  # 交叉验证
    clf.fit(train_data_x, train_data_y)
    print('最优c,g参数为:{0}'.format(clf.best_params_))
    svm_model = clf.best_estimator_
    return svm_model
# Perform classification procedures and output classification accuracy
def model_process(svm_model, test_data_x, test_data_y):
    p_lable = svm_model.predict(test_data_x)
    # 精确度为 生产者精度  召回率为 用户精度
    print('总体精度为 : {}'.format(accuracy_score(test_data_y, p_lable)))
    print('混淆矩阵为 :\n {}'.format(confusion_matrix(test_data_y, p_lable)))
    print('kappa系数为 :\n {}'.format(cohen_kappa_score(test_data_y, p_lable)))
    matric = confusion_matrix(test_data_y, p_lable)
    # output the accuracy of each category
    for category in range(np.max(test_data_y)):
        # add 0.0 to keep the float type of output
        precise = (matric[category, category] + 0.0) / np.sum(matric[category, :])
        recall = (matric[category, category] + 0.0) / np.sum(matric[:, category])
        f1_score = 2 * (precise * recall) / (recall + precise)
        print(
            '类别{}的生产者、制图(recall)精度为{:.4}  用户（precision）精度为{:.4}  F1 score 为{:.4} '.format(category + 1, precise, recall,
                                                                                          f1_score))
def open_txt_film(filepath):
    # open the film
    if os.path.exists(filepath):
        with open(filepath, mode='r') as f:
            train_data_str = np.loadtxt(f, delimiter=' ')
            print('训练（以及测试）数据的行列数为{}'.format(train_data_str.shape))
            return train_data_str
    else:
        print('输入txt文件路径错误，请重新输入文件路径')

def main():
    # read the train data from txt film
    train_file_path = r'E:\CSDN\data1\train.txt'
    train_data = open_txt_film(train_file_path)
    # read the predict data from txt film
    test_file_path = r'E:\CSDN\data1\test.txt'
    test_data = open_txt_film(test_file_path)

    # data normalization for svm training and testing dataset
    scaler = preprocessing.MinMaxScaler().fit(train_data[:, :-1])
    train_data[:, :-1] = scaler.transform(train_data[:, :-1])
    test_data[:, :-1] = scaler.transform(test_data[:, :-1])

    # conversion the unit,and the dimension to 1d
    train_data_y = train_data[:, -1:].astype('int')
    train_data_y = train_data_y.reshape(len(train_data_y))
    train_data_x = train_data[:, :-1]

    test_data_x = test_data[:, :-1]
    test_data_y = test_data[:, -1:].astype('int')
    test_data_y = test_data_y.reshape(len(test_data_y))
    model = grid_find(train_data_x,train_data_y)
    # keep the same scale of the train data
    model_process(model, test_data_x, test_data_y)


if __name__ == "__main__":
    # remember the beginning time of the program
    start_time = datetime.datetime.now()
    print("start...%s" % start_time)

    main()

    # record the running time of program with the unit of minutes
    end_time = datetime.datetime.now()
    sub_time_days = (end_time - start_time).days
    sub_time_minutes = (end_time - start_time).seconds / 60.0
    print("The program is last %s days , %s minutes" % (sub_time_days, sub_time_minutes))
