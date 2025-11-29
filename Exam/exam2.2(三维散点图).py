import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置Pandas显示选项（确保文本表格清晰无截断）
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1500)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# 读取数据
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

# 根据手肘图，假设最优簇数为 4
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(rfm_features_scaled)

# 将聚类结果添加到原始数据中
user_features['Cluster'] = kmeans.labels_

# 定义用户标签
user_labels = {
    0: '高价值活跃用户',
    1: '低价值不活跃用户',
    2: '中等价值普通用户',
    3: '潜在高价值用户'
}

# 将用户标签添加到原始数据中
user_features['User_Label'] = user_features['Cluster'].map(user_labels)

# 绘制三维散点图（保持原功能）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster in range(4):
    cluster_data = user_features[user_features['Cluster'] == cluster]
    ax.scatter(cluster_data['Recency'], cluster_data['Frequency'], cluster_data['Monetary'], label=user_labels[cluster])

ax.set_xlabel('最近购买天数')
ax.set_ylabel('购买次数')
ax.set_zlabel('总消费金额')
ax.set_title('用户分群结果（三维散点图）')
ax.legend()
plt.show()

print("="*120)
print("用户分群结果汇总")
print("="*120)
# 1. 输出前20条用户分群详情（避免输出过多，按需调整）
user_cluster_detail = user_features[['CustomerID', 'Cluster', 'User_Label', 'Recency', 'Frequency', 'Monetary']].head(20).copy()
# 数值格式化（保留2位小数，更整洁）
user_cluster_detail['Recency'] = user_cluster_detail['Recency'].round(2)
user_cluster_detail['Frequency'] = user_cluster_detail['Frequency'].round(2)
user_cluster_detail['Monetary'] = user_cluster_detail['Monetary'].round(2)

print("\n前20条用户分群详情：")
print(user_cluster_detail.to_string(index=False))  # 无索引文本表格输出

# 2. 输出各分群的用户数量统计
cluster_count = user_features['User_Label'].value_counts().reset_index()
cluster_count.columns = ['用户标签', '用户数量']
print("\n各分群用户数量统计：")
print(cluster_count.to_string(index=False))

# 3. 输出各分群的核心指标均值（帮助理解分群特征）
cluster_metrics = user_features.groupby('User_Label').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()
cluster_metrics['Recency'] = cluster_metrics['Recency'].round(2)
cluster_metrics['Frequency'] = cluster_metrics['Frequency'].round(2)
cluster_metrics['Monetary'] = cluster_metrics['Monetary'].round(2)
print("\n各分群核心指标均值：")
print(cluster_metrics.to_string(index=False))

print("="*120)