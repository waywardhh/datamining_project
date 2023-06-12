import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import data

result_label = pd.read_csv("customer_label.csv")
data = data.get_data()
columns_to_keep = ['X1','X3',  'X15', 'X16', 'X17', "X32", "X33", "X34", "X35", "X36", "X28", "X29", "X31"]
select_data = data.loc[0:10000, columns_to_keep]
for i in select_data.columns:
    select_data[i].fillna(select_data[i].mean(),inplace=True)
X = select_data[columns_to_keep].values
filter_result_label = result_label.loc[0:10000,['label']]
filter_result_label['0'] = filter_result_label['label'].map({1: 1, 0: 0})
filter_result_label['1'] = filter_result_label['label'].map({0: 1, 1: 0})
the_need_filter_result_label = filter_result_label.loc[:,["0",'1']]
y_need = np.asarray(the_need_filter_result_label)
y = filter_result_label["label"].values

def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:,column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))
transformer = SelectKBest(score_func=multivariate_pearsonr, k=3)
Xt_pearson = transformer.fit_transform(X, y)
print(transformer.scores_)
#['X3', 'X2', 'X15', 'X16', 'X17', 'X43', 'X42', "X40", 'X38', 'X44', "X45", "X46", "X47", "X4","X32","X33"]
features_to_scale = columns_to_keep
#使用StandardScaler进行Z-score标准化
scaler = StandardScaler()
select_data[features_to_scale] = scaler.fit_transform(select_data[features_to_scale])
Y = np.asmatrix(y_need)
X = select_data[columns_to_keep].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier(n_neighbors=50)
estimator.fit(X_train, y_train)
scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
predictions = estimator.predict(X_test)
print("The average accuracy is {0:.1f}%".format(average_accuracy))
print("F-score: {0:.2f}".format(f1_score(predictions, y_test)))
avg_scores = []
all_scores = []
parameter_values = list(range(1, 200)) # Include 20
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)
plt.plot(parameter_values,avg_scores, '-o')
plt.show()