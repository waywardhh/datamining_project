import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
import sys, numpy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
import data
sys.modules["scipy.random"] = numpy.random
import pybrain
sys.modules[__name__] = pybrain
from sklearn.preprocessing import StandardScaler
#数据处理，特征选取
result_label = pd.read_csv("customer_label.csv")
data = data.get_data()
columns_to_keep = ['X1', 'X3', 'X15', 'X16', 'X17', "X32", "X33", "X34", "X35", "X36", "X28", "X29", "X31", "X39",
                   "X37"]
select_data = data.loc[0:10000, columns_to_keep]
for i in select_data.columns:
    select_data[i].fillna(select_data[i].mean(), inplace=True)
X = select_data[columns_to_keep].values
filter_result_label = result_label.loc[0:10000, ['label']]
filter_result_label['0'] = filter_result_label['label'].map({0: 1, 1: 0})
filter_result_label['1'] = filter_result_label['label'].map({1: 1, 0: 0})
the_need_filter_result_label = filter_result_label.loc[:, ["0", '1']]
y_need = np.asarray(the_need_filter_result_label)
y = filter_result_label["label"].values

#利用皮尔逊（Pearson）相关系数对特征进行处理
def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:, column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))


transformer = SelectKBest(score_func=multivariate_pearsonr, k=3)
Xt_pearson = transformer.fit_transform(X, y)
print(transformer.scores_)

#使用StandardScaler进行Z-score标准化
features_to_scale = columns_to_keep
scaler = StandardScaler()
select_data[features_to_scale] = scaler.fit_transform(select_data[features_to_scale])
Y = np.asmatrix(y_need)
X = select_data[columns_to_keep].values
#主成分分析
pca = PCA(n_components=15)
Xd = pca.fit_transform(X)
np.set_printoptions(precision=3, suppress=True)
print(pca.explained_variance_ratio_)
#创建模型进行预测并评价
X_train, X_test, y_train, y_test = \
    train_test_split(Xd, Y, train_size=0.9)
from pybrain.datasets import SupervisedDataSet

training = SupervisedDataSet(Xd.shape[1], Y.shape[1])
for i in range(X_train.shape[0]):
    training.addSample(X_train[i], y_train[i])
testing = SupervisedDataSet(Xd.shape[1], Y.shape[1])
for i in range(X_test.shape[0]):
    testing.addSample(X_test[i], y_test[i])
from pybrain.tools.shortcuts import buildNetwork

net = buildNetwork(Xd.shape[1], 2, Y.shape[1], bias=True)
from pybrain.supervised.trainers import BackpropTrainer

trainer = BackpropTrainer(net, training, learningrate=0.01,
                          weightdecay=0.01)
trainer.trainEpochs(epochs=20)
predictions = trainer.testOnClassData(dataset=testing)
print(predictions)
y_test_need = np.asarray(y_test).argmax(axis=1)
accuracy = np.mean(y_test_need == predictions) * 100
print("The accuracy is {0:.1f}%".format(accuracy))
from sklearn.metrics import f1_score
print("F-score: {0:.2f}".format(f1_score(predictions,y_test_need)))

