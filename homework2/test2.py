import pandas as pd
import numpy as np

# 创建DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Henry", "Ivy", "Jack",
             "Alice", "Lily", "Mike", "Nancy", "Oscar", "Peter", "Queen", "Robert", "Steve", "Tom"],
    "Age": [25, 30, None, 45, -1, 33, 28, None, 40, 29,
            25, 31, 150, 22, 36, None, 27, 30, 45, None],
    "Gender": ["F", "M", "Tale", "female", "F", None, "F", "male", "F", "Unknown",
               "F", "F", "Tale", "Female", "F", "M", "Tale", "F", None, "M"],
    "Amount": [100, 200, 150, None, 300, 250, 400, -1, 150, 220,
               100, None, 9999, 260, 210, 230, 240, None, 300, 180]
}

# 创建原始数据的副本用于比较
df_original = pd.DataFrame(data)
df = df_original.copy()

print("原始数据:")
print(df.head(10))  # 只显示前10行
print("\n原始数据信息:")
print(df.info())
print("\n原始数据描述统计:")
print(df.describe())

# 1. 缺失值处理
print("\n" + "="*50)
print("1. 缺失值处理")
print("="*50)

# a) 找出各列缺失值数量
missing_values = df.isnull().sum()
print("各列缺失值数量:")
print(missing_values)

# b) 将Age中的缺失值用该列的中位数填充，将Amount中的缺失值用该列的均值填充
# 先计算中位数和均值（排除异常值）
age_median = df[(df['Age'] > 0) & (df['Age'] < 100)]['Age'].median()
amount_mean = df[(df['Amount'] > 0) & (df['Amount'] < 2000)]['Amount'].mean()

print(f"\nAge中位数: {age_median}")
print(f"Amount均值: {amount_mean}")

# 使用正确的方法填充缺失值
df = df.assign(
    Age=df['Age'].fillna(age_median),
    Amount=df['Amount'].fillna(amount_mean)
)

# 2. 数据清洗
print("\n" + "="*50)
print("2. 数据清洗")
print("="*50)

# a) 去掉Name前后的空格，并去重
df['Name'] = df['Name'].str.strip()
print(f"Name列去重前有 {len(df)} 行")
df = df.drop_duplicates(subset=['Name'], keep='first')
print(f"Name列去重后有 {len(df)} 行")

# b) 将Gender统一为小写的"male"和"female"（其它值统一为缺失）
# 首先标准化现有的值
gender_mapping = {
    'M': 'male',
    'male': 'male',
    'F': 'female',
    'female': 'female',
    'Female': 'female'
}

def standardize_gender(gender):
    if pd.isna(gender):
        return np.nan
    return gender_mapping.get(gender, np.nan)

df['Gender'] = df['Gender'].apply(standardize_gender)

print("\nGender列标准化后的值分布:")
print(df['Gender'].value_counts(dropna=False))

# 3. 异常值处理
print("\n" + "="*50)
print("3. 异常值处理")
print("="*50)

# a) 将Age中小于0或大于100的值设为缺失并用均值填充
age_mean = df[(df['Age'] > 0) & (df['Age'] < 100)]['Age'].mean()
print(f"Age均值(排除异常值): {age_mean}")

# 创建Age列的掩码
age_mask = (df['Age'] < 0) | (df['Age'] > 100)
df.loc[age_mask, 'Age'] = np.nan
df['Age'] = df['Age'].fillna(age_mean)

# b) 将Amount中小于0或大于2000的值设为缺失并用中位数填充
amount_median = df[(df['Amount'] > 0) & (df['Amount'] < 2000)]['Amount'].median()
print(f"Amount中位数(排除异常值): {amount_median}")

# 创建Amount列的掩码
amount_mask = (df['Amount'] < 0) | (df['Amount'] > 2000)
df.loc[amount_mask, 'Amount'] = np.nan
df['Amount'] = df['Amount'].fillna(amount_median)

# 4. 数据检查
print("\n" + "="*50)
print("4. 数据检查")
print("="*50)
print("清洗后的数据:")
print(df.head(10))  # 只显示前10行
print("\n清洗后的数据信息:")
print(df.info())
print("\n清洗后的数据描述统计:")
print(df.describe())

# 比较清洗前后的数据
print("\n" + "="*50)
print("清洗前后对比")
print("="*50)
print("原始数据形状:", df_original.shape)
print("清洗后数据形状:", df.shape)

print("\n原始数据缺失值统计:")
print(df_original.isnull().sum())

print("\n清洗后数据缺失值统计:")
print(df.isnull().sum())

print("\n原始数据Age列统计:")
print(f"最小值: {df_original['Age'].min()}, 最大值: {df_original['Age'].max()}, 均值: {df_original['Age'].mean():.2f}")

print("\n清洗后数据Age列统计:")
print(f"最小值: {df['Age'].min()}, 最大值: {df['Age'].max()}, 均值: {df['Age'].mean():.2f}")

print("\n原始数据Amount列统计:")
print(f"最小值: {df_original['Amount'].min()}, 最大值: {df_original['Amount'].max()}, 均值: {df_original['Amount'].mean():.2f}")

print("\n清洗后数据Amount列统计:")
print(f"最小值: {df['Amount'].min()}, 最大值: {df['Amount'].max()}, 均值: {df['Amount'].mean():.2f}")