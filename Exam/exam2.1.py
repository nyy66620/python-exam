import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 设置Pandas显示选项（确保表格完整无截断）
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1500)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# 读取数据
df = pd.read_csv('cleaned_Online_Retail.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')

#2.1用Apriori或FP-Growth算法挖掘频繁项集（最小支持度=0.01），输出前5条强关联规则
# 将数据转换为适合 Apriori 算法的格式
basket = (df.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack(fill_value=0) > 0)

# 使用 Apriori 算法挖掘频繁项集，启用低内存模式
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True, low_memory=True)

# 获取项集数量
num_itemsets = basket.shape[1]

# 生成关联规则
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1, num_itemsets=num_itemsets)

# 按置信度降序排序，输出前 5 条规则
top_5_rules = rules.sort_values(by='confidence', ascending=False).head(5)

print('前 5 条强关联规则：')
print(top_5_rules)