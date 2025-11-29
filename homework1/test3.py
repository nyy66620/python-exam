import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 生成随机成绩数据（0-100分之间的100个成绩）
np.random.seed(42)  # 设置随机种子以确保结果可重现
scores = np.random.normal(75, 15, 100)  # 均值为75，标准差为15的正态分布
scores = np.clip(scores, 0, 100)  # 确保分数在0-100范围内

# 创建图表和子图
plt.figure(figsize=(10, 6))

# 绘制直方图，设置不同颜色的柱状条
n, bins, patches = plt.hist(scores, bins=10, range=(0, 100),
                            edgecolor='black', linewidth=1.2, alpha=0.7)

# 设置不同颜色
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99FF',
          '#FF6B6B', '#4ECDC4', '#C7F464', '#FF6F61', '#6B5B95']
for i, patch in enumerate(patches):
    patch.set_facecolor(colors[i % len(colors)])

# 设置标题和轴标签
plt.title('成绩分布直方图', fontsize=16, fontweight='bold')
plt.xlabel('分数段', fontsize=12)
plt.ylabel('人数', fontsize=12)

# 设置x轴刻度
plt.xticks(np.arange(0, 101, 10))

# 添加网格
plt.grid(axis='y', alpha=0.3)

# 在柱状图上添加数值标签
for i, v in enumerate(n):
    if v > 0:  # 只在有数据的柱子上添加标签
        plt.text(bins[i] + 5, v + 0.5, str(int(v)),
                ha='center', va='bottom', fontweight='bold')

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 打印基本统计信息
print(f"成绩数据统计:")
print(f"平均分: {np.mean(scores):.2f}")
print(f"最高分: {np.max(scores):.2f}")
print(f"最低分: {np.min(scores):.2f}")
print(f"标准差: {np.std(scores):.2f}")