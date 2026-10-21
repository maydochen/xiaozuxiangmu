"""
目标：使用随机森林模型训练tfidf向量化后的文本分类

步骤：
1. 导入数据
2. 实例化sklearn的tfidf类
3. 拿到词向量
4. 实例化RandomForestClassifier
5. 训练
6. 模型评估
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from config import Config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from tqdm import tqdm
import os
import pickle


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
y= data['label'].tolist()
# 4. 实例化RandomForestClassifier

## 最优超参数: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}
#最佳交叉验证准确率: 0.5463
best_params = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}
model = RandomForestClassifier(**best_params)
##训练输入，测试输入，训练label，测试label
# X= X[:1000,:]
# y = y[:1000]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=22)

# 5. 训练
###5.1 超参优化
# # 设置超参数网格
# # 针对高维数据，重点调整 max_features 和 min_samples_split 来控制过拟合
# param_grid = {
#     'n_estimators': [100, 200, 300],          # 树的数量
#     'max_depth': [10, 20, 30, None],          # 树的最大深度
#     'min_samples_split': [2, 5, 10],          # 内部节点再划分所需最小样本数
#     'min_samples_leaf': [1, 2, 4],            # 叶子节点最少样本数
#     'max_features': ['sqrt', 'log2', 0.3]     # 高维数据关键：每次分裂考虑的最大特征数
# }
#
# #  初始化 GridSearchCV
# # cv=5 表示5折交叉验证, scoring='accuracy' 优化准确率, verbose=1 打印进度
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
#     verbose=1,
#     n_jobs=-1,
#     return_train_score=True
# )
#
# #  执行网格搜索
# print("开始网格搜索...")
# grid_search.fit(X_train, y_train)
#
# #  输出结果
# print(f"最佳参数: {grid_search.best_params_}")
# print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")



#
# 训练
for i in tqdm(range(1),desc="正在训练:"):
    model.fit(X_train,y_train)
# model.fit(X,y)

# 6. 模型评估
print(f"模型准确率：{accuracy_score(y_test, model.predict(X_test),)}")
print(f"模型精确率：{precision_score(y_test, model.predict(X_test),average='micro')}")
print(f"模型召回率：{recall_score(y_test, model.predict(X_test),average='micro')}")
print(f"模型f1：{f1_score(y_test, model.predict(X_test),average='micro')}")

# 7. 模型保存
## 创建目录
os.makedirs(config.model_path, exist_ok=True)
with open(config.model_path + "/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(config.model_path + "/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)