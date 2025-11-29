import numpy as np
import matplotlib.pyplot as plt
# 数据
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([52, 54, 61, 63, 66, 72, 74, 78])
def calculate_rss(beta0, beta1, x, y):
    # 计算预测值
    y_pred = beta0 + beta1 * x
    # 计算残差平方和
    rss = np.sum((y - y_pred) ** 2)
    return y_pred, rss
# 三组参数计算
params_list = [(50, 2), (48, 3), (55, 1.5)]
for beta0, beta1 in params_list:
    y_pred, rss = calculate_rss(beta0, beta1, x, y)
    print(f"当β₀={beta0}, β₁={beta1}时，残差平方和RSS={rss}")

# 最小二乘法计算最优β₀和β₁
x_mean = np.mean(x)
y_mean = np.mean(y)
beta1_opt = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
beta0_opt = y_mean - beta1_opt * x_mean
y_pred_opt, rss_opt = calculate_rss(beta0_opt, beta1_opt, x, y)
print(f"最优β₀={beta0_opt:.2f}, 最优β₁={beta1_opt:.2f}，此时残差平方和RSS={rss_opt:.2f}")

# 绘制图形
plt.scatter(x, y, color='blue', label='Actual Points')
# 绘制最优回归直线
plt.plot(x, y_pred_opt, color='red', label='Fitted Line')
# 绘制残差（竖线）
for xi, yi, ypi in zip(x, y, y_pred_opt):
    plt.plot([xi, xi], [yi, ypi], color='green', linestyle='--')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Score')
plt.title('Study Time vs Exam Score')
plt.legend()
plt.show()