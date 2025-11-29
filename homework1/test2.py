import pandas as pd
import numpy as np

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成模拟销售数据
num_records = 100
cities = ['北京', '上海', '广州']
sales = np.random.randint(50, 201, num_records)
employees = [f'E{str(i).zfill(3)}' for i in np.random.randint(1, 21, num_records)]

# 创建DataFrame
sales_data = pd.DataFrame({
    'city': np.random.choice(cities, num_records),
    'sales': sales,
    'employee': employees
})

print("生成的销售数据表:")
print(sales_data.head(10))
print('--------------------------------------------------------------------')

# 问题1: 按城市分组计算销售总额和平均销售额
city_stats = sales_data.groupby('city')['sales'].agg(['sum', 'mean']).round(2)
city_stats.columns = ['销售总额', '平均销售额']
print("1. 各城市销售统计:")
print(city_stats)
print('--------------------------------------------------------------------')

# 问题2: 添加城市平均销售额列并计算差值
# 首先创建城市平均销售额的映射
city_avg_map = sales_data.groupby('city')['sales'].mean().to_dict()
sales_data['city_avg_sales'] = sales_data['city'].map(city_avg_map)
sales_data['diff_from_avg'] = sales_data['sales'] - sales_data['city_avg_sales']

print("2. 添加城市平均销售额和差值后的数据:")
print(sales_data.head(10))
print('--------------------------------------------------------------------')

# 问题3: 筛选出销售额高于城市平均值的员工
high_performers = sales_data[sales_data['sales'] > sales_data['city_avg_sales']]
result = high_performers[['employee', 'city', 'sales']].reset_index(drop=True)

print("3. 销售额高于城市平均值的员工:")
print(result)
print(f"\n共有 {len(result)} 名员工销售额高于所在城市平均值")