import pandas as pd
import datetime as dt
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
from IPython.display import display

# 读取 Excel 文件
excel_file = pd.ExcelFile('Online Retail.xlsx')
# 获取所有表名
sheet_names = excel_file.sheet_names
print('表名：',sheet_names)
# 读取原表格的数据
df = excel_file.parse('Online Retail')
print('数据基本信息：')
df.info()
# 查看数据集行数和列数
rows, columns = df.shape
print('行数：',rows)
print('列数：',columns)

# 1.1 清洗缺失值和异常值
# 删除 CustomerID 缺失的记录
df = df.dropna(subset=['CustomerID'])
# 删除 Quantity 为负或 UnitPrice 为 0 的订单
df = df[(df['Quantity'] >= 0) & (df['UnitPrice'] > 0)]
# 将清洗后的数据保存为 CSV 文件
df.to_csv('cleaned_Online_Retail.csv', index=False)
print('数据基本信息：')
df.info()
# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))

# 设置Pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#1.2构造用户级特征
# 从 CSV 文件中读取清洗后的数据
df = pd.read_csv('cleaned_Online_Retail.csv')
# 明确日期格式进行转换
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')
# 设置参考日期
reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# 计算 RFM 指标
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'UnitPrice': lambda x: (x * df.loc[x.index, 'Quantity']).sum()  # Monetary
}).reset_index()

# 重命名列名
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# 定义晨间和夜间时间段
morning_start = dt.time(6, 0, 0)
morning_end = dt.time(11, 59, 59)
night_start = dt.time(20, 0, 0)
night_end = dt.time(5, 59, 59)

# 提取订单时间的小时信息
df['Hour'] = df['InvoiceDate'].dt.hour

# 标记订单是否在晨间或夜间
df['Morning_Order'] = df['Hour'].apply(lambda x: morning_start <= dt.time(x, 0, 0) <= morning_end)
df['Night_Order'] = df['Hour'].apply(lambda x: night_start <= dt.time(x, 0, 0) or dt.time(x, 0, 0) <= night_end)

# 计算每个用户的晨间和夜间订单占比
user_time_preference = df.groupby('CustomerID').agg({
    'Morning_Order': 'mean',
    'Night_Order': 'mean'
}).reset_index()

# 重命名列名
user_time_preference.columns = ['CustomerID', 'Morning_Order_Percentage', 'Night_Order_Percentage']

# 将 RFM 指标和购买时段偏好合并到用户特征表
user_features = pd.merge(rfm, user_time_preference, on='CustomerID')

print('用户级特征基本信息：')
user_features.info()

# 查看数据集行数和列数
rows, columns = user_features.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('用户级特征全部内容信息：')
    print(user_features.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('用户级特征前几行内容信息：')
    print(user_features.head().to_csv(sep='\t', na_rep='nan'))

#1.3 构造商品级特征
# 计算商品被购买频次
item_purchase_frequency = df.groupby('StockCode').size().reset_index(name='Purchase_Frequency')

# 计算每个商品的平均订单量
item_avg_order_quantity = df.groupby('StockCode')['Quantity'].mean().reset_index(name='Average_Order_Quantity')

# 将商品被购买频次和平均订单量合并到商品特征表
item_features = pd.merge(item_purchase_frequency, item_avg_order_quantity, on='StockCode')

print('商品级特征基本信息：')
item_features.info()

# 查看数据集行数和列数
rows, columns = item_features.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('商品级特征全部内容信息：')
    print(item_features.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('商品级特征前几行内容信息：')
    print(item_features.head().to_csv(sep='\t', na_rep='nan'))