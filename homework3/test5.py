import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 创建模拟的California Housing数据集
np.random.seed(42)
n_samples = 1000

# 特征名称（与真实California Housing数据集相同）
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']

# 生成特征数据
data = {
    'MedInc': np.abs(np.random.normal(3.0, 1.5, n_samples)),      # 收入中位数
    'HouseAge': np.abs(np.random.normal(30, 12, n_samples)),      # 房龄
    'AveRooms': np.abs(np.random.normal(5, 1.5, n_samples)),      # 平均房间数
    'AveBedrms': np.abs(np.random.normal(1, 0.3, n_samples)),     # 平均卧室数
    'Population': np.abs(np.random.normal(1500, 600, n_samples)), # 人口
    'AveOccup': np.abs(np.random.normal(3, 0.8, n_samples)),      # 平均占用率
    'Latitude': np.random.uniform(32, 42, n_samples),             # 纬度
    'Longitude': np.random.uniform(-124, -114, n_samples)         # 经度
}

df = pd.DataFrame(data)
X = df.values

# 创建真实系数（模拟真实房价关系）
true_coef = np.array([0.8, 0.15, 0.25, -0.1, -0.02, -0.08, 0.12, 0.1])
true_intercept = 0.3

# 生成目标变量（房屋中位价格）
y = X.dot(true_coef) + true_intercept + np.random.normal(0, 0.3, n_samples)
df['MedHouseVal'] = y

print("问题1：前5行数据展示")
print("=" * 50)
print(df.head())
print("\n数据集形状:", df.shape)
print("\n特征名称:", feature_names)
print("\n数据描述:")
print(df.describe())

# 问题2：拆分数据集并建立多元线性回归模型
print("\n" + "=" * 60)
print("问题2：建立多元线性回归模型")
print("=" * 60)

# 准备特征变量X和目标变量y
X = df[feature_names].values
y = df['MedHouseVal'].values

# 将数据集拆分为训练集和测试集（8:2比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 建立多元线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算评估指标
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n模型评估结果:")
print(f"决定系数 R²: {r2:.4f}")
print(f"均方误差 MSE: {mse:.4f}")
print(f"均方根误差 RMSE: {np.sqrt(mse):.4f}")

# 问题3：输出回归系数并解释两个自变量的影响
print("\n" + "=" * 60)
print("问题3：回归系数分析")
print("=" * 60)

# 输出模型的回归系数（包括截距）
print("回归系数（斜率）:")
print("-" * 30)
for i, feature in enumerate(feature_names):
    print(f"{feature:15}: {model.coef_[i]:.4f}")

print(f"\n截距 (Intercept): {model.intercept_:.4f}")

# 选择两个自变量进行详细解释
print("\n" + "=" * 60)
print("两个自变量的影响分析")
print("=" * 60)

# 选择两个有代表性的变量进行分析
selected_features = ['MedInc', 'AveRooms']

print("\n1. 收入中位数 (MedInc) 对房价的影响:")
print(f"   回归系数: {model.coef_[feature_names.index('MedInc')]:.4f}")
print("   解释: 收入中位数的系数为正数，表明收入水平与房价呈正相关关系。")
print("         当其他因素保持不变时，收入中位数每增加1个单位，房屋中位价格预计上涨约{:.4f}个单位。".format(
    model.coef_[feature_names.index('MedInc')]))
print("         这符合经济常识，收入越高的地区通常房价也越高。")

print("\n2. 平均房间数 (AveRooms) 对房价的影响:")
print(f"   回归系数: {model.coef_[feature_names.index('AveRooms')]:.4f}")
print("   解释: 平均房间数的系数为正数，表明房屋大小与房价呈正相关关系。")
print("         当其他因素保持不变时，平均房间数每增加1个单位，房屋中位价格预计上涨约{:.4f}个单位。".format(
    model.coef_[feature_names.index('AveRooms')]))
print("         这符合直觉，房间数越多通常意味着房屋面积越大，价格越高。")

# 显示所有变量的影响方向总结
print("\n" + "=" * 60)
print("所有变量的影响方向总结")
print("=" * 60)

for i, feature in enumerate(feature_names):
    direction = "正相关" if model.coef_[i] > 0 else "负相关"
    print(f"{feature:15}: {direction} (系数: {model.coef_[i]:.4f})")

# 可视化结果
plt.figure(figsize=(12, 5))

# 子图1：实际值 vs 预测值散点图
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.grid(True, alpha=0.3)

# 子图2：特征重要性条形图
plt.subplot(1, 2, 2)
feature_importance = np.abs(model.coef_)
plt.barh(feature_names, feature_importance)
plt.xlabel('特征重要性（系数绝对值）')
plt.title('特征重要性')
plt.tight_layout()

plt.show()

print("\n程序执行完成！")