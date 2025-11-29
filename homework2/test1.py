import numpy as np
from scipy.spatial.distance import pdist
#定义数据对象
x=np.array([2,4,3,6,8,2])
y=np.array([1,4,2,7,5,3])

#将数据对象堆叠成矩阵形式
X=np.vstack([x,y])
print(X)

euclidean_distance=pdist(X,'euclidean')
print(f'欧氏距离：{euclidean_distance[0]:.4f}')

manhattan_distance=pdist(X,'cityblock')
print(f'哈曼顿距离：{manhattan_distance[0]:.4f}')

chebyshev_distance=pdist(X,'chebyshev')
print(f'切比雪夫距离：{chebyshev_distance[0]:.4f}')

minkowski_distance=pdist(X,'minkowski',p=3)
print(f'闵可夫斯基距离(p=3)：{minkowski_distance[0]:.4f}')

cosine_distance=1-pdist(X,'cosine')
print(f'余弦相似性：{cosine_distance[0]:.4f}')