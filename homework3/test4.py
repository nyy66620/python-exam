import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 数据准备
# 房屋面积（自变量）
X = np.array([[1500], [1800], [2400], [3000], [3500], [4000], [4500]])
# 房屋价格（因变量）
y = np.array([245, 290, 315, 400, 475, 550, 600])

# 将数据集划分为训练集（80%）和测试集（20%），random_state 固定随机种子以保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 模型训练
# 创建一元线性回归模型并拟合训练集数据
model = LinearRegression()
model.fit(X_train, y_train)

# 得到回归系数 β₀（截距）和 β₁（斜率）
beta0 = model.intercept_
beta1 = model.coef_[0]
print(f"回归系数 β₀ = {beta0:.4f}，β₁ = {beta1:.4f}")
print(f"回归方程为：y = {beta0:.4f} + {beta1:.4f}x")

# 3. 模型评估
# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算评估指标：MAE（平均绝对误差）、MSE（均方误差）、RMSE（均方根误差）
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE = {mae:.4f}")
print(f"MSE = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")

# 4. 可视化
# 绘制回归直线（基于整个数据集的范围）
x_line = np.array([[X.min()], [X.max()]])
y_line = model.predict(x_line)

plt.figure(figsize=(10, 6))
# 绘制训练集散点
plt.scatter(X_train, y_train, color='blue', label='Training Data')
# 绘制测试集散点
plt.scatter(X_test, y_test, color='green', label='Testing Data')
# 绘制回归直线
plt.plot(x_line, y_line, color='red', label='Regression Line')

# 展示测试集的预测值与真实值，并标出误差（用竖线表示）
for x_val, y_true, y_pred_val in zip(X_test, y_test, y_pred):
    plt.plot([x_val, x_val], [y_true, y_pred_val], color='orange', linestyle='--')

plt.xlabel('House Area (square feet)')
plt.ylabel('House Price (thousand dollars)')
plt.title('House Area vs House Price - Linear Regression')
plt.legend()
plt.show()