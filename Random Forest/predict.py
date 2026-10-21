"""
目标：导入保存的模型，做预测

步骤：
1. 导入模型
2. 预测
"""
import pickle
from config import Config
import jieba
import time

# 步骤：
# 1. 导入模型
config = Config()
with open(config.model_path + "/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open(config.model_path + "/rf_model.pkl", "rb") as f:
    model = pickle.load(f)
# 导入标签定义
with open(config.label_path, 'r', encoding='utf-8') as f:
    ind2label = f.read().split('\n')

def predict(data):
    """
    预测函数封装，给定输入字典（包含text字段），返回
    {
    "text",
    "pred_class", # 预测结果
    "duration # 耗时
    }
    """
    t1 = time.time()

    # 2. 预测
    title = data["text"]
    # 向量化
    vector = tfidf_vectorizer.transform([' '.join(jieba.lcut(title)[:30])])
    # 预测
    result = model.predict(vector)[0]

    category_map = {
        "finance": "财经",
        "realty": "房产",
        "stocks": "股票",
        "education": "教育",
        "science": "科技",
        "society": "社会",
        "politics": "政治",
        "sports": "体育",
        "game": "游戏",
        "entertainment": "娱乐"
    }

    return {
        "text": data['text'],
        "pred_class": ind2label[result],
        "duration": time.time() - t1
    }


if __name__ == '__main__':
    text_list = ["俄罗斯新型洲际弹道导弹试射失败","巴基斯坦否认本-拉登藏在本国",
                 "日本幼儿园表扬用“真努力”而非“真优秀”",
                 "福布斯发布最贵购物榜：富豪4亿美元买岛屿居首",
                 "神舟13LED宽屏独显轻薄本只卖2799元",
                 "男子赌输扑克牌裸奔又裸泳",
                 "厨师下班遭五名大汉持刀围砍",
                 "劫匪因裤子被拉下露出内裤告警察",
                 "不老妖星一战写传奇 他被年轮绑住双腿却仍值尊敬",
                 "孙红雷笑言最欣赏林志玲胸部与美臀(图)"]
    for text in text_list:
        result = predict({"text":text})
        print(f"{text}的结果是{result}")