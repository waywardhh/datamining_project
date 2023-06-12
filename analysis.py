import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
data_filename = "customer.csv"
result_label = pd.read_csv("customer_label.csv")
data = pd.read_csv(data_filename)
print(data.columns)
the_need_man = data[data.X1 == "先生"]
#男女对比
the_need_sex_data = data['X1'].value_counts()
the_need_sex_data.index.name = "性别"
the_need_sex_data.name = '性别'
the_need_sex_data.plot.pie(title ='性别对比图')
print(the_need_sex_data)
#男女年龄对比
the_six_age_contrast_data = data.groupby(['X2', 'X1']).size().unstack()
the_six_age_contrast_data.columns.name = '性别'
the_six_age_contrast_data.index.name = "年龄"
print(the_six_age_contrast_data )
the_six_age_contrast_data.plot(title = '男女年龄统计',ylabel = "数量")


#细分市场分布图
the_market_data = data.groupby(['X5']).size()
the_costomer_star_data.plot(kind="pie", title="细分市场分布图")

#基于男女年龄的消费数据
the_consumption_age_contrast_gender_data = data['X15'].groupby([data["X2"], data['X1']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = "性别"
the_consumption_age_contrast_gender_data.index.name = "年龄"
the_consumption_age_contrast_gender_data.plot(title = "男女年龄的消费数据",ylabel = '消费平均数')
#基于男女在网时长的消费数据
the_consumption_time_contrast_gender_data = data['X15'].groupby([data["X4"], data['X1']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = '性别'
the_consumption_time_contrast_gender_data.index.name = '在网时长'
the_consumption_time_contrast_gender_data.plot(title = "男女在网时长的消费数据",ylabel = "消费平均数")
#基于男女年龄的流量数据
the_consumption_age_contrast_gender_data = data['X16'].groupby([data["X2"], data['X1']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = "性别"
the_consumption_age_contrast_gender_data.index.name = "年龄"
the_consumption_age_contrast_gender_data.plot(title = "男女年龄的流量数据",ylabel = '消费平均数')
#基于男女在网时长的流量数据
the_consumption_time_contrast_gender_data = data['X16'].groupby([data["X4"], data['X1']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = '性别'
the_consumption_time_contrast_gender_data.index.name = '在网时长'
the_consumption_time_contrast_gender_data.plot( title = "男女在网时长的流量数据",ylabel = "消费平均数")
#基于男女年龄的通话数据
the_consumption_age_contrast_gender_data = data['X17'].groupby([data["X2"], data['X1']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = "性别"
the_consumption_age_contrast_gender_data.index.name = "年龄"
the_consumption_age_contrast_gender_data.plot(title = "男女年龄的通话数据",ylabel = '消费平均数')
#基于男女在网时长的通话数据
the_consumption_time_contrast_gender_data = data['X17'].groupby([data["X4"], data['X1']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = '性别'
the_consumption_time_contrast_gender_data.index.name = '在网时长'
the_consumption_time_contrast_gender_data.plot( title="男女在网时长的通话数据", ylabel="消费平均数")
#基于细分市场的年龄的消费数据
the_consumption_age_contrast_gender_data = data['X15'].groupby([data["X2"], data['X5']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = "细分市场"
the_consumption_age_contrast_gender_data.index.name = "年龄"
the_consumption_age_contrast_gender_data.plot(title = "细分市场年龄的消费数据",ylabel = '消费平均数',figsize = (15,15))
#基于细分市场在网时长的消费数据
the_consumption_time_contrast_gender_data = data['X15'].groupby([data["X4"], data['X5']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = '细分市场'
the_consumption_time_contrast_gender_data.index.name = '在网时长'
the_consumption_time_contrast_gender_data.plot(title = "细分市场在网时长的消费数据",ylabel = "消费平均数")
#基于细分市场年龄的流量数据
the_consumption_age_contrast_gender_data = data['X16'].groupby([data["X2"], data['X5']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = "细分市场"
the_consumption_age_contrast_gender_data.index.name = "年龄"
the_consumption_age_contrast_gender_data.plot(title = "细分市场年龄的流量数据",ylabel = '消费平均数')
#基于细分市场在网时长的流量数据
the_consumption_time_contrast_gender_data = data['X16'].groupby([data["X4"], data['X5']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = '细分市场'
the_consumption_time_contrast_gender_data.index.name = '在网时长'
the_consumption_time_contrast_gender_data.plot( title = "细分市场在网时长的流量数据",ylabel = "消费平均数")
#基于细分市场年龄的通话数据
the_consumption_age_contrast_gender_data = data['X17'].groupby([data["X2"], data['X5']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = "细分市场"
the_consumption_age_contrast_gender_data.index.name = "年龄"
the_consumption_age_contrast_gender_data.plot(title = "细分市场年龄的通话数据",ylabel = '消费平均数')
#基于细分市场在网时长的通话数据
the_consumption_time_contrast_gender_data = data['X17'].groupby([data["X4"], data['X5']]).mean().unstack()
the_consumption_age_contrast_gender_data.columns.name = '细分市场'
the_consumption_time_contrast_gender_data.index.name = '在网时长'
the_consumption_time_contrast_gender_data.plot( title="细分市场在网时长的通话数据", ylabel="消费平均数")
plt.show()
test_data = data.groupby([data['X1'], data['X5']]).size().unstack()
print(test_data)
#潜在用户的数据
the_deta_consumption_data1 = data["X6"] - data["X7"]
the_deta_consumption_data2 = data['X7'] - data['X8']
the_deta_consumption_data3 = data["X18"] - data["X19"]
the_deta_consumption_data4 = data["X21"] - data["X22"]
the_potenial_costomer_data = data[(the_deta_consumption_data2 > 0) | (the_deta_consumption_data2 > 0)]
the_potenial_costomer_data = the_potenial_costomer_data[(the_deta_consumption_data3 >0)|(the_deta_consumption_data4 >0)]
the_potenial_costomer_data = the_potenial_costomer_data[data["X3"] <= 3]
print(len(the_potenial_costomer_data))
#流失用户的数据
the_lost_costomer_data = data[(the_deta_consumption_data2 < 0) | (the_deta_consumption_data2 < 0)]
the_lost_costomer_data = the_lost_costomer_data[(the_deta_consumption_data3 < 0)|(the_deta_consumption_data4 < 0)]
print(len(the_lost_costomer_data))