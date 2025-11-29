from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. 加载乳腺癌数据集
# 该数据集包含乳腺癌的特征和标签（恶性/良性）
data = load_breast_cancer()
X, y = data.data, data.target

# 2. 划分训练集和测试集
# test_size=0.2 表示测试集占比 20%，train_size=0.8 表示训练集占比 80%
# random_state=42 固定随机种子，确保结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 数据预处理（标准化）
# 标准化使特征均值为 0，标准差为 1，有助于逻辑回归等模型收敛和性能提升
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 逻辑回归模型与超参数寻优
# 定义要搜索的超参数网格，C 是正则化强度的倒数，C 越小正则化越强
# solver 选择 'liblinear'，它能处理小数据集且支持 L1、L2 正则化
# penalty 分别尝试 L1 和 L2 正则化
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2']
}
# GridSearchCV 用于网格搜索超参数，cv=5 表示 5 折交叉验证
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# 输出最佳超参数和交叉验证最佳得分
print("最佳超参数:", grid_search.best_params_)
print("交叉验证最佳准确率:", grid_search.best_score_)

# 5. 在测试集上评估最佳模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# 输出测试集上的准确率、分类报告和混淆矩阵
print("测试集准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))