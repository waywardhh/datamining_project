from mlxtend.frequent_patterns import apriori
import pandas as pd
import data

data_filename = "customer.csv"
results = pd.read_csv(data_filename)
result_label = pd.read_csv("customer_label.csv")
data = data.get_data()
the_need_data = data.drop(["X1", 'user_id', "X5"], axis=1)
for i in the_need_data.columns:
    the_need_data[i].fillna(0, inplace=True)

#有意义的（0，1）类别特征选出作为我们的新数据集并处理
the_filter_data = the_need_data.loc[0:10000, ["X24", "X25", "X27", "X28", "X29", "X30", "X31", "X38", "X39",
                                              "X40", "X41", "X42", "X43", "校园用户", "集团用户", "大众用户",
                                              "农村用户", "先生", "女士"]]
the_filter_data = the_filter_data.astype('bool')
transaction = []
for i in range(len(the_filter_data)):
    transaction.append([str(the_filter_data.values[i, j]) for j in range(len(the_filter_data.columns))])
transactions = pd.DataFrame(transaction).replace({'True': 1, 'False': 0})
transactions.columns = ["本网宽带用户", "异网宽带用户", "宽带激活", "宽带签约", "终端签约", "话费签约", "套餐签约",
                        "5G流量", "终端类型",
                        "低消保费用户", "换机", "居住地5G", "工作地5G", "校园用户", "集团用户", "大众用户", "农村用户",
                        "先生", "女士"]
#设置支持度为0.3，导入模型
frequent_itemsets = apriori(transactions, min_support=0.3, use_colnames=True)
from mlxtend.frequent_patterns import association_rules

# 设置最小提升度为1
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)
