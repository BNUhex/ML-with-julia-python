import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# 读取CSV文件
df = pd.read_csv('train.csv')

# 获取特征和标签
X = df.drop(columns=['defects'])  # 假设 'label' 是你的标签列
y = df['defects']

# 使用RandomUnderSampler进行欠抽样
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# 创建新的DataFrame
balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df['defects'] = y_resampled

# 将结果保存到新的CSV文件中
balanced_df.to_csv('balanced_data.csv', index=False)
