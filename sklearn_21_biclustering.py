# 双聚类
import numpy as np
data = np.arange(100).reshape(10, 10)
rows = np.array([0, 2, 3])[:, np.newaxis]
columns = np.array([1, 2])
print(data[rows, columns])
