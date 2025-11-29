import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

# 读取数据
df = pd.read_csv('cleaned_Online_Retail.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')

# 3.1 构建二分类模型预测用户 30 天内复购概率
# 1. 定义时间分割点
max_date = df['InvoiceDate'].max()
prediction_window = 30
train_end_date = max_date - dt.timedelta(days=prediction_window)

# 2. 划分训练期和预测期数据
train_data = df[df['InvoiceDate'] <= train_end_date]
prediction_data = df[df['InvoiceDate'] > train_end_date]

# 3. 构建目标变量
all_users = train_data['CustomerID'].unique()
repeat_users = prediction_data['CustomerID'].unique()
target = pd.DataFrame({'CustomerID': all_users})
target['Repeat_30d'] = target['CustomerID'].apply(lambda x: 1 if x in repeat_users else 0)

# 4. 构建训练特征
# 4.1 训练期 RFM 特征
train_rfm = train_data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (train_end_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'UnitPrice': lambda x: (x * train_data.loc[x.index, 'Quantity']).sum()
}).reset_index()
train_rfm.columns = ['CustomerID', 'Train_Recency', 'Train_Frequency', 'Train_Monetary']

# 4.2 训练期购买时段偏好
morning_start = dt.time(6, 0, 0)
morning_end = dt.time(11, 59, 59)
night_start = dt.time(20, 0, 0)
night_end = dt.time(5, 59, 59)

train_data = train_data.copy()
train_data.loc[:, 'Hour'] = train_data['InvoiceDate'].dt.hour
train_data.loc[:, 'Morning_Order'] = train_data['Hour'].apply(
    lambda x: morning_start <= dt.time(x, 0, 0) <= morning_end
)
train_data.loc[:, 'Night_Order'] = train_data['Hour'].apply(
    lambda x: night_start <= dt.time(x, 0, 0) or dt.time(x, 0, 0) <= night_end
)

time_preference = train_data.groupby('CustomerID').agg({
    'Morning_Order': 'mean',
    'Night_Order': 'mean'
}).reset_index()
time_preference.columns = ['CustomerID', 'Morning_Percent', 'Night_Percent']

# 4.3 其他用户行为特征
item_diversity = train_data.groupby('CustomerID')['StockCode'].nunique().reset_index(name='Item_Diversity')
avg_quantity = train_data.groupby('CustomerID')['Quantity'].mean().reset_index(name='Avg_Quantity')

# 合并所有特征
features = pd.merge(train_rfm, time_preference, on='CustomerID')
features = pd.merge(features, item_diversity, on='CustomerID')
features = pd.merge(features, avg_quantity, on='CustomerID')

# 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.drop('CustomerID', axis=1))
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns[1:])
features_scaled_df['CustomerID'] = features['CustomerID'].values

# 5. 合并特征和目标变量
data = pd.merge(features_scaled_df, target, on='CustomerID')

# 6. 划分特征和目标
X = data.drop(['CustomerID', 'Repeat_30d'], axis=1)
y = data['Repeat_30d']

# 获取特征名列表
feature_names = X.columns.tolist()

# 7. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. 构建和训练随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 9. 预测和评估模型
y_pred_proba = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
auc = roc_auc_score(y_test, y_pred_proba)
print('AUC:', auc)
print('分类报告：')
print(classification_report(y_test, y_pred))

# 3.2 输出关键特征重要性排名（Gini重要性）
print("\n=== 特征重要性排名（Gini重要性）===")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# 可视化特征重要性
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Gini重要性')
plt.title('随机森林特征重要性排名')
plt.gca().invert_yaxis()  # 重要性从高到低显示
plt.tight_layout()
plt.show()

# 3.3 针对不同用户分群提出数据驱动的营销策略
# 使用完整的特征数据进行聚类
rfm_features = features[['Train_Recency', 'Train_Frequency', 'Train_Monetary']]
scaler_rfm = StandardScaler()
rfm_features_scaled = scaler_rfm.fit_transform(rfm_features)

kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(rfm_features_scaled)

# 将聚类结果添加到原始特征数据中
features_with_cluster = features.copy()
features_with_cluster['Cluster'] = cluster_labels

# 定义用户标签
user_labels = {
    0: '高价值活跃用户',
    1: '低价值不活跃用户',
    2: '中等价值普通用户',
    3: '潜在高价值用户'
}

features_with_cluster['User_Label'] = features_with_cluster['Cluster'].map(user_labels)

# 分析各用户群体的复购率
user_analysis = pd.merge(features_with_cluster[['CustomerID', 'User_Label']],
                         data[['CustomerID', 'Repeat_30d']], on='CustomerID')

repurchase_by_group = user_analysis.groupby('User_Label')['Repeat_30d'].agg(['mean', 'count']).round(3)
repurchase_by_group.columns = ['复购率', '用户数']
print("\n=== 各用户群体复购率分析 ===")
print(repurchase_by_group)

# 基于特征重要性和用户分群的营销策略
print("\n=== 数据驱动的营销策略建议 ===")

# 根据最重要的特征制定策略
top_features = feature_importance.head(3)['feature'].tolist()
print(f"最重要的三个特征: {top_features}")

marketing_strategies = {
    '高价值活跃用户': [
        '提供VIP专属折扣和优先购买权',
        '邀请参加新品预览和专属活动',
        '提供个性化商品推荐和定制服务',
        '建立高级会员积分兑换体系'
    ],
    '低价值不活跃用户': [
        '发送高价值折扣券刺激复购',
        '推送热门商品和限时优惠信息',
        '开展用户召回活动，提供回归礼包',
        '分析购买历史，推荐相关商品'
    ],
    '中等价值普通用户': [
        '提供阶梯式优惠，鼓励提升消费',
        '推荐商品套餐和组合优惠',
        '开展积分奖励和兑换活动',
        '定期推送个性化促销信息'
    ],
    '潜在高价值用户': [
        '提供新用户专属优惠礼包',
        '优质客服一对一咨询服务',
        '推送热门和高评价商品推荐',
        '开展试用活动和体验课程'
    ]
}

# 针对不同用户群体的具体策略
for user_group, strategies in marketing_strategies.items():
    print(f"\n{user_group}营销策略:")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")

# 基于特征重要性的具体建议
print(f"\n=== 基于特征重要性的具体建议 ===")
top_feature = feature_importance.iloc[0]['feature']
print(f"最重要的特征 '{top_feature}' 建议:")

if 'Recency' in top_feature:
    print("- 针对近期未购买用户发送提醒和优惠")
    print("- 设置购买间隔提醒机制")
    print("- 开展定期回访和关怀活动")
elif 'Frequency' in top_feature:
    print("- 设置购买频次奖励计划")
    print("- 推荐订阅或定期购买服务")
    print("- 提供多买多优惠策略")
elif 'Monetary' in top_feature:
    print("- 设置消费金额阶梯优惠")
    print("- 提供大额订单专属服务")
    print("- 推荐高价值商品和套餐")
elif 'Morning' in top_feature or 'Night' in top_feature:
    print("- 在用户活跃时段推送营销信息")
    print("- 设置时段专属优惠活动")
    print("- 优化对应时段的客服安排")

# 保存模型预测结果和用户分群
all_predictions = rf.predict_proba(X)[:, 1]

result_df = pd.DataFrame({
    'CustomerID': data['CustomerID'],
    'Predicted_Probability': all_predictions,
    'Predicted_Class': (all_predictions > 0.5).astype(int),
    'Actual_Repeat': data['Repeat_30d'],
    'User_Label': features_with_cluster['User_Label'],
    'High_Risk_Churn': (all_predictions < 0.3).astype(int)  # 低复购概率用户标记为高流失风险
})

# 分析高流失风险用户分布
high_risk_analysis = result_df.groupby('User_Label')['High_Risk_Churn'].mean().round(3)
print(f"\n=== 各用户群体高流失风险比例 ===")
print(high_risk_analysis)

# 针对高流失风险用户的专项策略
print(f"\n=== 高流失风险用户专项策略 ===")
high_risk_users = result_df[result_df['High_Risk_Churn'] == 1]
if not high_risk_users.empty:
    print(f"发现 {len(high_risk_users)} 名高流失风险用户")
    print("建议立即采取以下措施:")
    print("1. 发送专属高价值优惠券")
    print("2. 进行客户满意度调研")
    print("3. 提供个性化商品推荐")
    print("4. 安排客服主动联系关怀")
else:
    print("未发现高流失风险用户")

# 保存完整结果
result_df.to_csv('model_prediction_results.csv', index=False)
print("\n预测结果已保存到 model_prediction_results.csv")

# 输出模型性能总结
print(f"\n=== 模型性能总结 ===")
print(f"AUC: {auc:.4f}")
accuracy = (y_pred == y_test).mean()
print(f"准确率: {accuracy:.4f}")
print(f"特征数量: {len(feature_names)}")
print(f"用户总数: {len(data)}")
print(f"复购用户比例: {data['Repeat_30d'].mean():.4f}")