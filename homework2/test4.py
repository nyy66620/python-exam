import pandas as pd

# 创建数据表1：Jan
jan_data = {
    'date': ['2025-01-01', '2025-01-02', '2025-01-01'],
    'city': ['Beijing', 'Beijing', 'Shanghai'],
    'clicks': [100, 120, 80]
}
jan_df = pd.DataFrame(jan_data)

# 创建数据表2：Feb
feb_data = {
    'date': ['2025-02-01', '2025-02-01'],
    'city': ['Beijing', 'Shenzhen'],
    'clicks': [150, 60],
    'device': ['mobile', 'desktop']
}
feb_df = pd.DataFrame(feb_data)

print("数据表1 - Jan:")
print(jan_df)
print("\n数据表2 - Feb:")
print(feb_df)

# 使用concat进行纵向拼接
# axis=0: 按行拼接（纵向）
# join='outer': 保留所有列，缺失位置用NaN填充
# ignore_index=True: 重新生成连续的行索引
concatenated_data = pd.concat(
    [jan_df, feb_df],
    axis=0,
    join='outer',
    ignore_index=True
)

print("\n拼接后的数据:")
print(concatenated_data)

# 详细解释
print("\n" + "="*60)
print("拼接结果说明:")
print("="*60)
print("1. 按行纵向拼接: 将两个DataFrame上下连接")
print("2. 保留所有列: Jan表有3列(date, city, clicks), Feb表有4列(多出device列)")
print("3. 缺失值处理: Jan表没有device列，所以前3行的device显示为NaN")
print("4. 重新索引: 使用ignore_index=True重新生成0-4的连续行索引")
print("5. 未排序: 保持原始数据的顺序，没有进行排序")
print("6. 未去重: 保留所有行，包括可能重复的数据")

# 验证结果
print("\n" + "="*60)
print("结果验证:")
print("="*60)
print(f"拼接后数据形状: {concatenated_data.shape}")
print(f"拼接后列名: {list(concatenated_data.columns)}")
print(f"行索引范围: {concatenated_data.index.tolist()}")
print("\n各列数据类型:")
print(concatenated_data.dtypes)