import os

import numpy as np
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建和当前文件夹同级的文件夹，即使存在也不会报错（exist_ok)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:  # python三引号允许一个字符串跨多行，字符串中可以包含换行符、制表符以及其他特殊字符。
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')

data = pd.read_csv(data_file)
print(data)

# ## 处理缺失值
#
# 注意，“NaN”项代表缺失值。
# [**为了处理缺失的数据，典型的方法包括*插值法*和*删除法*，**]
# 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。
# 在(**这里，我们将考虑插值法**)。
#
# 通过位置索引`iloc`，我们将`data`分成`inputs`和`outputs`，
# 其中前者为`data`的前两列，而后者为`data`的最后一列。
# 对于`inputs`中缺少的数值，我们用同一列的均值替换“NaN”项。
#

# In[3]:


inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

# [**对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别。**]
# 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”，
# `pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
# 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。
# 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
#

# In[4]:


inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# ## 转换为张量格式
#
# [**现在`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式。**]
# 当数据采用张量格式后，可以通过在 :numref:`sec_ndarray`中引入的那些张量函数来进一步操作。
#

# In[5]:


X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, y)

# ## 小结
#
# * `pandas`软件包是Python中常用的数据分析工具中，`pandas`可以与张量兼容。
# * 用`pandas`处理缺失的数据时，我们可根据情况选择用插值法和删除法。
#
# ## 练习
#
# 创建包含更多行和列的原始数据集。
#
# 1. 删除缺失值最多的列。
# 2. 将预处理后的数据集转换为张量格式。

# 1.1 统计每列的缺失值数量
na_counts = np.sum(data.isna(), axis=0)

# 1.2 找出缺失值数量最多的列标签
label = data.columns[np.argmax(na_counts)]

# 1.3 删除该列
data = data.drop(columns=label)

# 2. 转换为tensor
X = torch.tensor(data.iloc[:, :-1].values)  # values是一个属性
y = torch.tensor(data.iloc[:, -1].values)
