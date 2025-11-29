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


# 读取数据
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
rfm_features = user_features[['Recency', 'Frequency', 'Monetary']]

# 数据标准化
scaler = StandardScaler()
rfm_features_scaled = scaler.fit_transform(rfm_features)

# 使用手肘法确定最优的簇数
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_features_scaled)
    inertia.append(kmeans.inertia_)

# 绘制手肘图
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('手肘图')
plt.xlabel('簇数')
plt.xticks(rotation=45)
plt.ylabel('簇内误差平方和')
plt.show()