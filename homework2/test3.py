import pandas as pd

# 创建数据表1：Enrollments
enrollments_data = {
    'student_id': [1, 1, 2, 3],
    'course': ['DS', 'ML', 'DS', 'CV']
}
enrollments = pd.DataFrame(enrollments_data)

# 创建数据表2：Grades
grades_data = {
    'student_id': [1, 1, 2, 3],
    'course': ['DS', 'DS', 'DS', 'NLP'],
    'grade': [85, 90, 88, 92]
}
grades = pd.DataFrame(grades_data)

print("数据表1 - Enrollments:")
print(enrollments)
print("\n数据表2 - Grades:")
print(grades)

# 使用左连接合并数据
# how='left': 左连接，保留左表所有行
# on=['student_id', 'course']: 使用student_id和course作为连接键
merged_data = pd.merge(
    enrollments,
    grades,
    how='left',
    on=['student_id', 'course']
)

print("\n合并后的数据:")
print(merged_data)

# 详细解释
print("\n" + "="*50)
print("合并结果说明:")
print("="*50)
print("1. 左连接(left join): 保留Enrollments表的所有行")
print("2. 连接键: student_id 和 course")
print("3. 一对多关系: student_id=1, course='DS' 在Grades表中有2条记录，所以展开为2行")
print("4. 空值处理: 没有匹配的成绩显示为NaN")
print("5. 具体匹配情况:")
print("   - student_id=1, course='DS': 匹配到2个成绩(85, 90)")
print("   - student_id=1, course='ML': 没有匹配成绩 → NaN")
print("   - student_id=2, course='DS': 匹配到1个成绩(88)")
print("   - student_id=3, course='CV': 没有匹配成绩 → NaN")