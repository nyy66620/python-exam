import numpy as np
import matplotlib.pyplot as plt
# 1. 准备数据
x = np.array([2, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12])
y = np.array([8, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 27])
# 2. 计算回归系数
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
# 计算分子和分母，用于求解β₁
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
beta1 = numerator / denominator
beta0 = y_mean - beta1 * x_mean
# 输出回归方程参数
print(f"回归方程为：ŷ = {beta0:.4f} + {beta1:.4f}x")
# 3. 预测广告费用为10万元时的销售额
x_pred = 10
y_pred = beta0 + beta1 * x_pred
print(f"当广告费用为{ x_pred }万元时，销售额大约为{ y_pred:.2f}万元")
# 4. 绘制散点图和回归直线
plt.scatter(x, y, color='blue', label='Data Points')
# 生成用于绘制回归直线的x值
x_line = np.linspace(min(x), max(x), 100)
y_line = beta0 + beta1 * x_line
plt.plot(x_line, y_line, color='red', label='Regression Line')
plt.xlabel('Advertising Cost (x, ten thousand yuan)')
plt.ylabel('Sales (y, ten thousand yuan)')
plt.title('Advertising vs Sales')
plt.legend()
plt.show()
# 5. 解释广告费用对销售额的影响
print(f"从回归方程ŷ = {beta0:.4f} + {beta1:.4f}x可以看出，广告费用每增加1万元，销售额大约增加{beta1:.4f}万元，说明广告费用的增加对销售额有显著的正向促进作用。")