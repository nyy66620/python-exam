import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. 加载Diabetes数据集
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print("数据集形状:", X.shape)
print("特征名称:", diabetes.feature_names)
print("目标变量范围: [{:.2f}, {:.2f}]".format(y.min(), y.max()))

# 2. 选择单一特征bmi作为输入变量（bmi是第2个特征，索引为2）
bmi_index = 2  # bmi特征的索引
X_bmi = X[:, bmi_index].reshape(-1, 1)  # 选择bmi特征并重塑为二维数组

print(f"\n选择的特征: {diabetes.feature_names[bmi_index]}")
print(f"bmi特征统计: 均值={X_bmi.mean():.3f}, 标准差={X_bmi.std():.3f}")

# 3. 将数据集拆分为训练集和测试集（7:3，随机种子固定为42）
X_train, X_test, y_train, y_test = train_test_split(
    X_bmi, y, test_size=0.3, random_state=42
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 4. 使用不同阶数的多项式特征进行实验
degrees = [1, 2, 3, 4, 5]
results = []

print("\n" + "=" * 60)
print("多项式回归模型结果比较")
print("=" * 60)

# 存储每个模型的预测结果用于可视化
predictions = {}

for degree in degrees:
    # 生成多项式特征
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test_poly)
    predictions[degree] = y_pred

    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # 存储结果
    results.append({
        '阶数': degree,
        'R²分数': r2,
        '均方误差MSE': mse,
        '特征数量': X_train_poly.shape[1]
    })

    print(f"阶数 {degree}: R² = {r2:.4f}, MSE = {mse:.4f}, 特征数 = {X_train_poly.shape[1]}")

# 5. 输出结果表格并确定最佳模型
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("结果汇总表格")
print("=" * 60)
print(results_df.to_string(index=False))

# 确定最佳模型（R²最高或MSE最低）
best_by_r2 = results_df.loc[results_df['R²分数'].idxmax()]
best_by_mse = results_df.loc[results_df['均方误差MSE'].idxmin()]

print("\n" + "=" * 60)
print("模型性能分析")
print("=" * 60)
print(f"基于R²分数的最佳模型: 阶数 {best_by_r2['阶数']} (R² = {best_by_r2['R²分数']:.4f})")
print(f"基于MSE的最佳模型: 阶数 {best_by_mse['阶数']} (MSE = {best_by_mse['均方误差MSE']:.4f})")

# 明确说明哪个阶数的模型表现最准确
if best_by_r2['阶数'] == best_by_mse['阶数']:
    best_degree = best_by_r2['阶数']
    print(f"\n🎯 最准确的模型: 阶数 {best_degree}")
    print(f"   - R²分数: {best_by_r2['R²分数']:.4f}")
    print(f"   - 均方误差MSE: {best_by_mse['均方误差MSE']:.4f}")
else:
    # 如果R²和MSE选择的不一致，优先考虑R²
    best_degree = best_by_r2['阶数']
    print(f"\n🎯 最准确的模型: 阶数 {best_degree} (基于R²分数)")
    print(f"   - R²分数: {best_by_r2['R²分数']:.4f}")
    print(f"   - 均方误差MSE: {results_df[results_df['阶数'] == best_degree]['均方误差MSE'].values[0]:.4f}")

# 6. 可视化结果
plt.figure(figsize=(15, 10))

# 子图1: 不同阶数模型的拟合曲线
plt.subplot(2, 2, 1)
# 生成用于绘制平滑曲线的点
x_range = np.linspace(X_bmi.min(), X_bmi.max(), 100).reshape(-1, 1)
plt.scatter(X_test, y_test, alpha=0.6, label='测试数据', color='lightgray')

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_range_poly = poly.fit_transform(x_range)

    # 训练完整数据的模型用于绘制曲线
    model_plot = LinearRegression()
    X_bmi_poly = poly.fit_transform(X_bmi)
    model_plot.fit(X_bmi_poly, y)
    y_range_pred = model_plot.predict(x_range_poly)

    plt.plot(x_range, y_range_pred, label=f'阶数 {degree}', linewidth=2)

plt.xlabel('BMI特征')
plt.ylabel('疾病进展')
plt.title('不同阶数多项式回归拟合曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: R²分数比较
plt.subplot(2, 2, 2)
plt.plot(degrees, results_df['R²分数'], 'o-', linewidth=2, markersize=8)
plt.xlabel('多项式阶数')
plt.ylabel('R²分数')
plt.title('不同阶数的R²分数比较')
plt.grid(True, alpha=0.3)
for i, r2 in enumerate(results_df['R²分数']):
    plt.annotate(f'{r2:.3f}', (degrees[i], r2), textcoords="offset points", xytext=(0, 10), ha='center')

# 子图3: MSE比较
plt.subplot(2, 2, 3)
plt.plot(degrees, results_df['均方误差MSE'], 'o-', linewidth=2, markersize=8, color='red')
plt.xlabel('多项式阶数')
plt.ylabel('均方误差MSE')
plt.title('不同阶数的MSE比较')
plt.grid(True, alpha=0.3)
for i, mse in enumerate(results_df['均方误差MSE']):
    plt.annotate(f'{mse:.1f}', (degrees[i], mse), textcoords="offset points", xytext=(0, 10), ha='center')

# 子图4: 最佳模型的预测 vs 实际值
plt.subplot(2, 2, 4)
best_pred = predictions[best_degree]
plt.scatter(y_test, best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title(f'最佳模型(阶数{best_degree})预测 vs 实际值')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 分析过拟合现象
print("\n" + "=" * 60)
print("过拟合分析")
print("=" * 60)
print("随着多项式阶数增加，模型可能会出现过拟合：")
print("- 低阶(1-2): 可能欠拟合，无法捕捉复杂关系")
print("- 中阶(3): 通常是最佳平衡点")
print("- 高阶(4-5): 可能过拟合，在训练集表现好但测试集差")

# 计算训练集上的表现进行比较
print("\n训练集与测试集性能对比:")
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train_poly))
    test_r2 = r2_score(y_test, model.predict(X_test_poly))

    print(f"阶数 {degree}: 训练集R² = {train_r2:.4f}, 测试集R² = {test_r2:.4f}, 差距 = {train_r2 - test_r2:.4f}")