"""
目标：使用tf-idf 词频逆文档频率对新闻标题进行向量化

步骤：
1. 导入数据
2. 实例化sklearn的tfidf类
3. 拿到词向量
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from config import Config


# 步骤：
# 1. 导入数据
config = Config()

data = pd.read_csv(config.processing_training_data_path)
# print(data)

# 2. 实例化sklearn的tfidf类

## 读取中止词列表
with open(config.stop_words_path, 'r',encoding='utf-8') as f:
    stop_words = f.read().split('\n')
# print(stop_words)
tfidf = TfidfVectorizer(stop_words=stop_words)
# 3. 拿到词向量
X = tfidf.fit_transform(data['words'].tolist())
# print(X)
print(X.shape)

# x = tfidf.transform(["词汇 阅读 是 关键 08年 考研 暑期 英语 复习 全 指南"])


