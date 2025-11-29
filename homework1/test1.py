import pandas as pd
import numpy as np

# 第一部分：Series操作
fruit_dict = {'apple': 10, 'banana': 20, 'cherry': 30, 'date': 40}

# 1.1 创建Series并按字母顺序排列索引
fruit_series = pd.Series(fruit_dict)
fruit_series_sorted = fruit_series.sort_index()
print("1.1 按字母顺序排列索引的Series:")
print(fruit_series_sorted)
print('--------------------------------------------------------------------')

# 1.2 对Series中的每个元素加上5
fruit_series_plus_5 = fruit_series_sorted + 5
print("1.2 每个元素加上5的结果:")
print(fruit_series_plus_5)
print('--------------------------------------------------------------------')

# 1.3 计算Series中所有值的平方
fruit_series_squared = fruit_series_sorted ** 2
print("1.3 值的平方:")
print(fruit_series_squared)
print('--------------------------------------------------------------------')

# 第二部分：时间序列操作
# 创建时间序列，假设从2023-01-01开始，共5天
dates = pd.date_range('2025-01-01', periods=5)
values = [1, 2, np.nan, 4, 5]
time_series = pd.Series(values, index=dates)
print("原始时间序列:")
print(time_series)
print('--------------------------------------------------------------------')

# 2.1 前向填充缺失值
time_series_ffill = time_series.ffill()
print("2.1 前向填充后的时间序列:")
print(time_series_ffill)
print('--------------------------------------------------------------------')

# 2.2 按周重采样并计算每周平均值
# 注意：由于数据只有5天，可能跨越两周，重采样会按周分组
time_series_resampled = time_series_ffill.resample('W').mean()
print("2.2 按周重采样的平均值:")
print(time_series_resampled)
print('--------------------------------------------------------------------')

# 2.3 后向填充缺失值（使用原始时间序列）
time_series_bfill = time_series.bfill()
print("2.3 后向填充后的时间序列:")
print(time_series_bfill)