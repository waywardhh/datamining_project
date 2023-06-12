import pandas as pd
def get_data():
    data_filename = "customer.csv"
    results = pd.read_csv(data_filename)
    result_label = pd.read_csv("customer_label.csv")
    # 去除用户号码这列
    data = results.drop(["product_no"], axis=1)
    #添加新特征
    data['先生'] = data['X1'].map({'女士': 0, '先生': 1})
    data['女士'] = data['X1'].map({'女士': 1, '先生': 0})
    data['X1'] = data['X1'].map({'女士': 0, '先生': 1})
    data['校园用户'] = data['X5'].map({'校园用户': 1})
    data['集团用户'] = data['X5'].map({'集团用户': 1})
    data['大众用户'] = data['X5'].map({'大众用户': 1})
    data['农村用户'] = data['X5'].map({'农村用户': 1})
    data['超额流量平均值'] = (data['X18'] + data["X19"] + data["X20"]) / 3
    data['超额语音花费平均值'] = (data['X21'] + data["X22"] + data["X23"]) / 3
    the_need_data = data.drop(["X1", 'user_id', "X5"], axis=1)
    for i in the_need_data.columns:
        the_need_data[i].fillna(0, inplace=True)
    return data